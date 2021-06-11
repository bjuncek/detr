import torch
import torchvision
import torch.nn.functional as F
from torchvision.models.detection.transform import resize_boxes

from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

from .backbone import build_backbone
from .matcher import build_matcher
from .query_encoding import build_query_encoding, RectangleQueryEncoding
from .pool_module import build_pooling_module
from .detr import DETR
from .transformer import build_transformer
from datasets import transforms as T
from torchvision.transforms.functional import to_pil_image as toPIL

class DETRack(DETR):
    def __init__(
        self,
        backbone,
        transformer,
        query_encoder,
        pool_module,
        num_classes,
        num_queries,
        transform,
    ):

        # assert isinstance(query_encoder, RectangleQueryEncoding)
        self.image = []
        self.transform = transform
        super(DETRack, self).__init__(
            backbone,
            transformer,
            query_encoder,
            pool_module,
            num_classes,
            num_queries 
        )
    
    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device

        # transform boxes
        h, w = self.preprocessed_images.size()[-2:]
        boxes = boxes.to(device)
        boxes = resize_boxes(boxes, self.original_image_sizes[0], [h, w])
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
        query_emb = [{"boxes": boxes}]
        del boxes


        outputs = self(self.preprocessed_images, query_emb)
        del query_emb

        out_logits, out_bbox = outputs["pred_logits"].detach(), outputs["pred_boxes"].detach()
        del outputs
        
        prob = F.softmax(out_logits, -1)
        if out_logits.size()[-1] == 2:
            scores, _ = prob.max(-1)
        else:
            scores, _ = prob[..., :-1].max(-1)
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = self.original_image_sizes[0]
        img_h , img_w = torch.tensor([img_h]),torch.tensor([img_w])
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
        boxes = boxes * scale_fct

        del scale_fct, img_h, img_w
        del prob
        del out_logits
        
        return boxes.squeeze(0).detach(), scores.squeeze(0).detach()
        

    def detect(self, img):
        # does entire detection step;
        # this will be a major PITA
        pass

    def load_image(self, images):
        # the input image here is a tensor of 0-1s
        device = list(self.parameters())[0].device
        # images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]
        
        preprocessed_images, _ = [self.transform(toPIL(img), None) for img in images][0]
        self.preprocessed_images = preprocessed_images.unsqueeze(0).to(device)
 



def make_default_transforms():
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return T.Compose([T.RandomResize([800], max_size=1333), normalize,])



def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    num_classes = args.num_classes
    if args.add_class:
        num_classes += 1

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    query_encoder = build_query_encoding(args)
    pool_module = build_pooling_module(args)

    model = DETRack(
        backbone,
        transformer,
        query_encoder,
        pool_module,
        num_classes=num_classes,
        num_queries=args.num_queries,
        transform=make_default_transforms()
    )

    print("Loading model weights")
    weights = torch.load(args.detr_detect_model, map_location="cpu")
    model.load_state_dict(weights['model'])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num param: ", n_parameters)
    model.eval()


    return model