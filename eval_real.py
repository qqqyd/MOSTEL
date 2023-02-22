import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
from rec_utils import AttnLabelConverter
from rec_dataset import RawDataset, AlignCollate
from rec_model import Rec_Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(opt):
    """ model configuration """
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.use_rgb:
        opt.input_channel = 3
    model = Rec_Model(opt)

    # load model
    print('Loading pretrained model from %s' % opt.saved_model)
    rec_state_dict = torch.load(opt.saved_model, map_location=device)
    if len(rec_state_dict) == 1:
        rec_state_dict = rec_state_dict['recognizer']
    rec_state_dict = {k.replace('module.', ''): v for k, v in rec_state_dict.items()}
    model.load_state_dict(rec_state_dict)
    model = torch.nn.DataParallel(model).to(device)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    n_correct = 0
    norm_ED = 0
    length_of_data = 0

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list, image_gts in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob, gt in zip(image_path_list, preds_str, preds_max_prob, image_gts):
                img_name = img_name.split('/')[-1]
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if not opt.sensitive:
                    pred = pred.lower()
                    gt = gt.lower()
                length_of_data += 1
                if pred == gt:
                    n_correct += 1
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

        accuracy = n_correct / float(length_of_data) * 100
        norm_ED = norm_ED / float(length_of_data)

        print(f'{opt.image_folder}: Total {length_of_data}\t Acc {accuracy:0.3f}\t normalized_ED {norm_ED:0.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help='path to image_folder which contains text images')
    parser.add_argument('--gt_file', help='path to gt_file')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--use_rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    main(opt)

