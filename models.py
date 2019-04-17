import fastai
from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks.hooks import num_features_model, model_sizes
import torchvision
import pretrainedmodels
import glob
from utils import *

def onehot_enc(labels, n_class=10):
    onehot = torch.zeros([len(labels), n_class], dtype=torch.float32).to(device)
    #onehot.scatter_(dim=1, index=ids, src=1)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)
    return onehot

def get_backbone(config):
    arch = config.model.backbone
    #base_arch = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    #f = globals().get(arch)
    if arch == 'resnet18':
        backbone = models.resnet18
    elif arch == 'resnet34':
        backbone = models.resnet34
    elif arch == 'resnet50':
        backbone = models.resnet50
    elif arch == 'densenet121':
        backbone = models.densenet121
    return backbone

def get_body(config):
    backbone = get_backbone(config)
    body = create_body(backbone)
    return body

def get_loss_fn(config):
    loss = config.loss.name
    if loss == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss == 'PNLoss':
        loss_fn = PNLoss(p_threshold=config.loss.p_threshold, n_threshold=config.loss.n_threshold, unknow_class=config.loss.unknown_class)
    elif loss == 'ArcFace':
        radius = config.loss.radius
        margin = config.loss.margin
        loss_fn = ArcFaceLoss(radius=radius, margin=margin)
    elif loss == 'CosFace':
        radius = config.loss.radius
        margin = config.loss.margin
        loss_fn = CosFaceLoss(radius=radius, margin=margin)
    elif loss == 'MixLoss':
        radius = config.loss.radius
        margin = config.loss.margin
        loss_fn = MixLoss(radius=radius, margin=margin)
    return loss_fn

class ArcModule1(nn.Module):
    def __init__(self, in_features, out_features, radius=65, margin=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius = radius
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        #sin_th = (1.0 - cos_th**2) ** 0.5
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.radius
        return outputs


class CosSimCenters(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.centers)
        #nn.init.kaiming_normal_(self.centers)

    def forward(self, in_features):#, labels):
        cos_th = F.linear(F.normalize(in_features), F.normalize(self.centers))
        cos_th = cos_th.clamp(-1, 1)
        #theta = torch.acos(cos_th)
        #cos_th_m = torch.cos(theta+self.margin)
        return cos_th

    def pred(self, in_features):
        cos = self.forward(in_features)
        return torch.softmax(cos, dim=1)


class BinaryHead(nn.Module):
    def __init__(self, num_class=10008, emb_size = 2048, s = 16.0):
        super().__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)*self.s
        return logit


class CosHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_features = config.model.n_emb
        self.out_features = config.model.n_class

        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.BatchNorm1d(self.in_features*2),
                                  nn.Dropout(config.model.drop_rate/2),
                                  nn.Linear(self.in_features*2, 2048),
                                  nn.PReLU(),
                                  nn.BatchNorm1d(2048),
                                  nn.Dropout(config.model.drop_rate),
                                  nn.Linear(2048, self.out_features),
                                  )

        self.head = nn.Sequential(#AdaptiveConcatPool2d(),
                                  #Flatten(),
                                  nn.BatchNorm2d(self.in_features),
                                  nn.Dropout(config.model.drop_rate),
                                  Flatten(),
                                  nn.Linear(self.in_features*49, self.in_features),
                                  #nn.PReLU(),
                                  nn.BatchNorm1d(self.in_features),
                                  #nn.Dropout(config.model.drop_rate),
                                  #nn.Linear(2048, self.out_features),
                                  CosSimCenters(self.in_features, self.out_features)
                                  )

        for m in self.head.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(self.head)

    def cal_features(self, x):
        return self.forward(x)

    def forward(self, x):
        cos_th = self.head(x)
        return cos_th


class MixHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_features = config.model.n_emb
        self.out_features = config.model.n_class

        self.head = nn.Sequential(#AdaptiveConcatPool2d(),
                                  #Flatten(),
                                  nn.BatchNorm2d(self.in_features),
                                  nn.Dropout(config.model.drop_rate),
                                  Flatten(),
                                  nn.Linear(self.in_features*49, self.in_features),
                                  #nn.PReLU(),
                                  nn.BatchNorm1d(self.in_features),
                                  #nn.Dropout(config.model.drop_rate),
                                  #nn.Linear(2048, self.out_features),
                                  #CosSimCenters(self.in_features, self.out_features)
                                  )

        #self.bn1 = nn.BatchNorm1d(self.in_features)
        self.sphere_multi = CosSimCenters(self.in_features, self.out_features - 1)
        self.sphere_binary = nn.Linear(self.out_features - 1, 1)
        nn.init.xavier_normal_(self.sphere_binary.weight)

        for m in self.head.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(self.head)

        self.binary_classifier = nn.Linear(self.in_features, self.out_features)
        nn.init.xavier_normal_(self.binary_classifier.weight)

    def cal_features(self, x):
        return self.forward(x)

    def forward(self, x):
        ho = self.head(x)
        cos_th = self.sphere_multi(ho)
        bin_logits = self.sphere_binary(cos_th)
        #return torch.cat([cos_th, bin_logits], dim=1)
        return [cos_th, bin_logits]

def predict_mixhead(model, dl):
    model.eval()
    with torch.no_grad():
        cos_o, bin_o = [], []
        cnt = 0
        y_list = []
        for x, y in progress_bar(dl):
            x = torch.tensor(x).to(device)
            [cos_logits, bin_logits] = model(x)
            cos_o.append(cos_logits)
            bin_o.append(bin_logits)
            y_list.append(y)
            #cnt += 1
            #if cnt == 3:
            #    break

        cos_o = torch.cat(cos_o)
        bin_o = torch.cat(bin_o).view(-1)
        y = torch.cat(y_list).view(-1)

    return [cos_o, bin_o], y

def topk_mix(cos_logits, bin_logits):
        top5 = cos_logits.topk(5, 1)[1]
        bin = (torch.sigmoid(bin_logits) > 0.5).long() * 5004

        for row in range(len(bin)):
            if bin[row]:
                top5[row, 1:] = top5[row, :-1]
                top5[row, 0] = bin[row]
        return top5




class CosNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_emb = config.model.n_emb
        self.radius = config.model.radius
        self.n_class = config.model.n_class
        self.body = get_body(config)
        nf = num_features_model(nn.Sequential(*self.body.children())) * 2
        self.head = create_head(nf, self.n_emb, lin_ftrs=[1024], ps=config.model.drop_rate, concat_pool=True, bn_final=True)
        self.cos_sim = CosSimCenters(self.n_emb, self.n_class)
        self.unknown_classifier = nn.Linear(self.n_class, 1)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        cos_th = self.cos_sim(x)
        unknown_logits = self.unknown_classifier(x)
        return [cos_th, unknown_logits]

    def cal_features(self, x):
        x = self.body(x)
        x = self.head(x)
        x = F.normalize(x)
        return x

    def pred(self, in_images):
        cos = self.forward(in_images)
        cos = F.normalize(cos)
        return torch.softmax(cos, dim=1)


class CosNet1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_emb = config.model.n_emb
        self.radius = config.model.radius
        self.n_class = config.model.n_class
        self.body = get_body(config)
        nf = num_features_model(nn.Sequential(*self.body.children())) * 2
        self.head = create_head(nf, self.n_emb, lin_ftrs=[1024], ps=config.model.drop_rate, concat_pool=True, bn_final=True)
        self.cos_sim = CosSimCenters(self.n_emb, self.n_class)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        x = self.cos_sim(x)
        return x

    def cal_features(self, x):
        x = self.body(x)
        x = self.head(x)
        x = F.normalize(x)
        return x

    def pred(self, in_images):
        cos = self.forward(in_images)
        cos = F.normalize(cos)
        return torch.softmax(cos, dim=1)

    '''
    def pred1(self, in_images, target):
        cos = self.forward(in_images)
        cos_th = cos[0]
        cos_th_m = cos[1]
        onehot = onehot_enc(target, n_class=cos_th.shape[-1])
        #if target.dim() == 1:
        #    target = target.unsqueeze(-1)
        #onehot = torch.zeros(cos_th.size()).to(device)
        #onehot.scatter_(1, target, 1)
        logits = onehot * cos_th_m + (1.0 - onehot) * cos_th
        logits = logits * self.radius     #rescale
        return torch.softmax(logits, dim=1)
    '''

class ArcFaceLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', radius=60, margin=0.5):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.radius = radius
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    #@weak_script_method
    def forward1(self, input, target):
        cos_th = input[0]
        cos_th_m = input[1]
        onehot = onehot_enc(target, n_class=cos_th.shape[-1])
        #if target.dim() == 1:
        #    target = target.unsqueeze(-1)
        #onehot = torch.zeros(cos_th.size()).to(device)
        #onehot.scatter_(1, target, 1)
        logits = onehot * cos_th_m + (1.0 - onehot) * cos_th
        logits = logits * self.radius     #rescale
        return F.cross_entropy(logits, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, input, target):
        cos_th = input[0]
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m        #cos(theta + margin)
        cos_th_m = torch.where(cos_th > self.threshold, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.threshold
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        onehot = onehot_enc(target, n_class=cos_th.shape[-1])
        #if target.dim() == 1:
        #    target = target.unsqueeze(-1)
        #onehot = torch.zeros(cos_th.size()).to(device)
        #onehot.scatter_(1, target, 1)
        logits = onehot * cos_th_m + (1.0 - onehot) * cos_th
        logits = logits * self.radius     #rescale
        return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class CosFaceLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', radius=60, margin=0.4):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.radius = radius
        self.margin = margin

    #@weak_script_method
    def forward(self, input, target):
        cos_th = input[0]
        target_onehot = onehot_enc(target, n_class=cos_th.shape[-1])
        logits = self.radius * (cos_th * (1 - target_onehot) + (cos_th - self.margin) * target_onehot)
        return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


def focal_loss(logits, targets, weights=1.0, use_focal=True, gamma=2.0):
    ce = F.cross_entropy(logits, targets.long(), reduction='none')
    softmax = F.softmax(logits, dim=-1)
    target_probs = softmax.gather(dim=1, index=targets.long().view(-1, 1))
    focal = 1.0
    if use_focal:
        focal = torch.pow(1 - target_probs, gamma)
    focal_loss = focal * weights * ce.view(-1, 1)
    return focal_loss.mean()


def bin_focal_loss(logits, targets, weights=1.0, use_focal=True, gamma=2.0, reduction='mean'):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    #ce = F.binary_cross_entropy(logits, targets, reduction='none')
    softmax = F.softmax(logits, dim=-1)
    target_probs = softmax.gather(dim=1, index=targets.long())
    focal = 1.0
    if use_focal:
        focal = torch.pow(1 - target_probs, gamma)
    focal_loss = focal * weights * ce
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss



def bce(logits, target, n_hard=20, margin=0.4, use_focal=True):
    target_onehot = onehot_enc(target, 5005)
    #probs = torch.sigmoid(logits)
    #probs = (probs * (1 - target_onehot) + torch.relu(probs - margin) * target_onehot)
    logits = (logits * (1 - target_onehot) + (logits - margin) * target_onehot)
    #loss = F.binary_cross_entropy_with_logits(input, target_onehot, reduce=False)
    loss = bin_focal_loss(logits, target_onehot, gamma=3.0, use_focal=use_focal, reduction='none')
    loss = loss.topk(n_hard, 1)[0]
    #target_value = loss.gather(1, target.view(-1, 1))
    #sec_avg = (loss.topk(n_hard, largest=True, sorted=False, dim=1)[0][:, -sec_len:].sum(1) - target_value).mean() / sec_len
    #ohem_loss = torch.relu(target_value - sec_avg + margin)
    #return ohem_loss.mean()
    return loss.mean()


def bce1(input, target, n_hard=None):
    if n_hard is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')
        return loss
    else:
        target_onehot = onehot_enc(target, 5005)
        loss = F.binary_cross_entropy_with_logits(input, target_onehot, reduce=False)
        value, index = loss.topk(n_hard, dim=1, largest=True, sorted=False)

        return value.mean()


class MixLoss(nn.Module):
    def __init__(self, radius=60, margin=0.4, model=None, unknow_class=5004):
        super().__init__()
        self.cos_loss = CosFaceLoss(radius=radius, margin=margin)#, reduction='none')
        #self.cos_loss = ArcFaceLoss(radius=radius, margin=margin)#, reduction='none')
        self.unknown = unknow_class
        self.cnt = -1
        self.avg_loss_known = 20.0
        self.n_hard = 200
        self.step = 0
        self.sec_len = 20
        self.margin = 0.6
        self.n_thresh = 0.01
        self.model = model

    def forward(self, logits, target):
        self.step += 1
        self.n_hard = int(linear_schedule(self.step, [2000, 20, 0, 20000]))
        #acc = acc_with_unknown(logits, target)
        #map5 = mapk_with_unknown(logits, target)

        cos_logits = logits[0]
        bin_logits = logits[1].view_as(target)

        #for known, cos_loss
        logits_known = cos_logits[target!=self.unknown]
        target_known = target[target!=self.unknown]
        loss_known = self.avg_loss_known
        if len(target_known):
            loss_known = self.cos_loss(logits_known, target_known)
            self.avg_loss_known = self.avg_loss_known * 0.9 + loss_known * 0.1

        loss_bin = F.binary_cross_entropy_with_logits(bin_logits, (target==self.unknown).float())
        loss_bin *= 10

        #l2 regularization
        reg_loss = torch.tensor(0, dtype=torch.float32).to(device)
        if self.model:
            for param in self.model.parameters():
                reg_loss += (param ** 2).sum()
            reg_loss *= 3e-5

        loss = loss_known + loss_bin + reg_loss

        self.cnt += 1
        if self.cnt % 1000 == 0:
            print(f'loss_bin={loss_bin}, loss_known={loss_known}, reg_loss={reg_loss} \n')

        return loss

    def forward1(self, logits, target):
        self.step += 1
        self.n_hard = int(linear_schedule(self.step, [2000, 20, 0, 20000]))

        cos_logits = logits[0]
        bin_logits = logits[1]

        #for known, cos_loss
        logits_known = cos_logits[target!=self.unknown]
        target_known = target[target!=self.unknown]
        loss_known = self.avg_loss_known
        if len(target_known):
            loss_known = self.cos_loss(logits_known, target_known)
            self.avg_loss_known = self.avg_loss_known * 0.9 + loss_known * 0.1

            target_known_value = logits_known.gather(1, target_known.view(-1, 1))
            sec_avg = (logits_known.topk(self.n_hard, largest=True, sorted=False, dim=1)[0][:, -self.sec_len:].sum(1) - target_known_value).mean() / self.sec_len
            ohem_loss = torch.relu(sec_avg - target_known_value + self.margin)
            ohem_loss = ohem_loss.mean() * 10

        #for all = known + new_whale, binary_loss
        bce_loss = 0.0
        bce_loss = bce(bin_logits, target, self.n_hard)
        bce_loss *= 10

        #penalize the new_whale examples which are too high
        logits_unknown = cos_logits[target==self.unknown]
        n_loss = 0.0
        if len(logits_unknown):
            softmax_unknown = F.softmax(logits_unknown, -1)
            softmax_unknown = softmax_unknown.max(dim=1)[0]
            n_loss = F.relu(softmax_unknown - self.n_thresh).mean()
            n_loss *= 100

        #loss = self.avg_loss_known# + ohem_loss# + bce_loss
        #loss = loss_known + ohem_loss + bce_loss
        loss = loss_known + bce_loss + n_loss

        self.cnt += 1
        if self.cnt % 400 == 0:
            print(f'n_hard={self.n_hard}, bce_loss={bce_loss}, loss_known={loss_known}, ohem_loss={ohem_loss}, n_loss={n_loss}\n')

        return loss


class MixLoss1(nn.Module):
    def __init__(self, radius=60, margin=0.4, unknow_class=5004):
        super().__init__()
        self.cos_face_loss = CosFaceLoss(radius=radius, margin=margin)
        self.unknown = unknow_class
        self.cnt = 0
        self.avg_loss_known = 20.0

    def forward(self, logits, target):
        unknown_logits = logits[:, 0]
        logits = logits[:, 1:]

        unknown_target = (target==self.unknown)
        #pred_unknown = torch.sigmoid(unknown_logits) > 0.5
        binary_loss = F.binary_cross_entropy_with_logits(unknown_logits.view_as(target), unknown_target.float())#, reduction='none')

        logits_known = logits[target!=self.unknown]
        target_known = target[target!=self.unknown]
        loss_known = self.avg_loss_known
        if len(target_known):
            loss_known = self.cos_face_loss(logits_known, target_known)
            self.avg_loss_known = self.avg_loss_known * 0.9 + loss_known * 0.1

        loss = loss_known + binary_loss * 10

        self.cnt += 1
        if self.cnt % 400 == 0:
            print(f'binary_loss={binary_loss}, loss_known={loss_known}\n')

        return loss


class PNLoss(nn.Module):
    def __init__(self, p_threshold, n_threshold=None, unknow_class=5004):
        super().__init__()
        self.set_threshold(p_threshold, n_threshold)
        self.unknown = unknow_class

    def set_threshold(self, p_thresh, n_thresh=None):
        self.p_thresh = p_thresh
        self.n_thresh = n_thresh
        if n_thresh is None:
            self.n_thresh = (1 - p_thresh) / 2

    def forward(self, logits, target):
        softmax = F.softmax(logits, 1)
        #split known and unknown
        p_target = target[target!=self.unknown]
        p_softmax = softmax[target!=self.unknown]
        #n_target = target[target==self.unknown]
        n_softmax = softmax[target==self.unknown]

        p_value = p_softmax.gather(1, p_target.view(-1, 1))
        # subtract one margin for p_value itself
        p_loss = F.relu(p_softmax + self.p_thresh - p_value).sum(dim=1) - self.p_thresh

        n_loss = F.relu(n_softmax - self.n_thresh).sum(dim=1)

        loss = torch.cat([p_loss, n_loss]).mean()
        #ploss = p_loss.mean()
        #nloss = n_loss.mean()

        return loss#, ploss, nloss



from utils import map5
class CalMap5Callback(fastai.callbacks.tracker.TrackerCallback):
    "Calculate Map score"
    def __init__(self, learn):
        super().__init__(learn)

    def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    #def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        preds_known = []
        preds_unknown = []
        targets = []
        self.learn.model.eval()
        with torch.no_grad():
            for k, (data, labels) in enumerate(self.learn.data.valid_dl):
                #print(k)
                #if k >=15: break
                #softmax = self.learn.model.pred(data)
                softmax, binary = self.learn.model(data)
                preds_known.append(softmax)
                preds_unknown.append(binary)
                targets.append(labels)
            preds_known = torch.cat(preds_known)
            preds_unknown = torch.cat(preds_unknown)
            targets = torch.cat(targets)
            #acc = accuracy_with_unknown([preds_known, preds_unknown], targets)
            #print(acc)
            map_score = mapk_with_unknown([preds_known, preds_unknown], targets)
            print(f'map score: {map_score}\n')


class ResumeEpoch(fastai.callbacks.tracker.TrackerCallback):
    "Resume epoch"
    def __init__(self, learn, resume_epoch):
        super().__init__(learn)
        self.resume = resume_epoch

    def on_train_begin(self, epochs:int, pbar:PBar, metrics:MetricFuncList)->None:
        "About to start learning."
        self.state_dict = {'epoch':self.resume, 'iteration':0, 'num_batch':0}
        self.state_dict.update(dict(n_epochs=epochs, pbar=pbar, metrics=metrics))
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)
        if self.state_dict['epoch'] != 0:
            self.state_dict['pbar'].first_bar.total -= self.state_dict['epoch']
            for cb in self.callbacks: cb.jump_to_epoch(self.state_dict['epoch'])


class ScoreboardCallback(fastai.callbacks.tracker.TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, scoreboard, monitor:str='val_score', mode:str='auto', config=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.prefix = config.model.backbone
        self.monitor = monitor
        self.config = config
        if isinstance(scoreboard, Scoreboard):
            self.scoreboard = scoreboard
        else:
            if isinstance(scoreboard, Path):
                self.scoreboard_file = scoreboard
            elif isinstance(scoreboard, str):
                if 'scoreboard' in scoreboard:
                    self.scoreboard_file = Path(scoreboard)
                else:
                    self.scoreboard_file = pdir.models/f'scoreboard-{scoreboard}.pkl'
            self.sb_len = config.scoreboard.len
            self.scoreboard = Scoreboard(self.scoreboard_file, self.sb_len, sort='dec')
        self.best_score = 0
        self.mode = mode
        self.operator = np.greater
        if monitor == 'val_loss':
            self.best_score = np.inf
            self.mode = 'min'
            self.operator = np.less
        self.patience = config.train.patience
        self.wait = 0

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch}', purge=False)
            print(f"Loaded {self.name}_{epoch}")
        except: print(f'Model {self.name}_{epoch} not found.')

    #def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.monitor == 'val_loss':
            score = self.get_monitor_value()
        else:  #map score
            _, score = self.learn.validate(self.learn.data.valid_dl)
        print(f'score = {score}')

        # early stopping
        if score is None: return
        #if score > self.best_score:
        if self.operator(score, self.best_score):
            self.best_score,self.wait = score,0
        else:
            self.wait += 1
            print(f'wait={self.wait}, patience={self.patience}')
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                return {"stop_training":True}

        #scoreboard
        #if len(self.scoreboard) == 0 or score > self.scoreboard[-1][0]:
        if not self.scoreboard.is_full() or self.operator(score, self.scoreboard[-1]['score']):
            store_file = f'{self.prefix}-{epoch}'
            save_path = self.learn.save(store_file, return_path=True)
            plog.info('$$$$$$$$$$$$$ Good score {} at training step {} $$$$$$$$$'.format(score, epoch))
            plog.info(f'save to {save_path}')
            update_dict = {'score': score.item(),
                           'epoch': epoch,
                           'timestamp': start_timestamp,
                           'config': self.config,
                           'file': save_path
                           }
            self.scoreboard.update(update_dict)

    def on_train_end(self, **kwargs):
        "Load the best model."
        #print('tmp saving model to linshi')
        #self.learn.save(f'linshi')
        if len(self.scoreboard):
            self.learn.load(self.scoreboard[0]['file'].name[:-4], purge=False)

