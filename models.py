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
    elif loss == 'ArcFace':
        radius = config.loss.radius
        margin = config.loss.margin
        loss_fn = ArcFaceLoss(radius=radius, margin=margin)
    elif loss == 'CosFace':
        radius = config.loss.radius
        margin = config.loss.margin
        loss_fn = CosFaceLoss(radius=radius, margin=margin)
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


class CosHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_features = config.model.pars.n_emb
        self.out_features = config.model.pars.n_class

        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.BatchNorm1d(self.in_features*2),
                                  nn.Dropout(config.model.pars.drop_rate/2),
                                  nn.Linear(self.in_features*2, 2048),
                                  nn.PReLU(),
                                  nn.BatchNorm1d(2048),
                                  nn.Dropout(config.model.pars.drop_rate),
                                  nn.Linear(2048, self.out_features),
                                  )

        self.head = nn.Sequential(#AdaptiveConcatPool2d(),
                                  #Flatten(),
                                  nn.BatchNorm2d(self.in_features),
                                  nn.Dropout(config.model.pars.drop_rate),
                                  Flatten(),
                                  nn.Linear(self.in_features*49, self.in_features),
                                  #nn.PReLU(),
                                  nn.BatchNorm1d(self.in_features),
                                  #nn.Dropout(config.model.pars.drop_rate),
                                  #nn.Linear(2048, self.out_features),
                                  CosSimCenters(self.in_features, self.out_features)
                                  )

        for m in self.head.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        #nn.init.kaiming_normal_(self.head)

    def cal_features(self, x):
        x = self.head(x)
        return F.normalize(x)

    def forward(self, x):
        x = self.head(x)
        return x



class CosNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_emb = config.model.pars.n_emb
        self.radius = config.model.pars.radius
        self.n_class = config.model.pars.n_class
        self.body = get_body(config)
        nf = num_features_model(nn.Sequential(*self.body.children())) * 2
        self.head = create_head(nf, self.n_emb, lin_ftrs=[1024], ps=config.model.pars.drop_rate, concat_pool=True, bn_final=True)
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

    def forward(self, cos_th, target):
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
    def forward(self, cos_th, target):
        onehot = onehot_enc(target, n_class=cos_th.shape[-1])
        logits = self.radius * (cos_th * (1 - onehot) + (cos_th - self.margin) * onehot)
        return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


from utils import map5
class CalMap5Callback(fastai.callbacks.tracker.TrackerCallback):
    "Calculate Map score"
    def __init__(self, learn):
        super().__init__(learn)

    #def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        preds = []
        targets = []
        self.learn.model.eval()
        with torch.no_grad():
            for k, (data, labels) in enumerate(self.learn.data.valid_dl):
                #print(k)
                #if k >=15: break
                #softmax = self.learn.model.pred(data)
                softmax = self.learn.model(data)
                preds.append(softmax)
                targets.append(labels)
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            map_score = mapkfast(preds, targets)
            print(f'map score: {map_score}\n')


class ScoreboardCallback(fastai.callbacks.tracker.TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, scoreboard, monitor:str='val_score', mode:str='auto', patience=30, config=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.prefix = 'densenet121'
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
        self.patience = patience
        self.wait = 0

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
    #def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
        "Compare the value monitored to its best score and maybe save the model."
        val_result = self.learn.validate(self.learn.data.valid_dl)
        val_loss = val_result[0]
        val_score = val_result[-1]
        if self.monitor == 'val_loss':
            score = val_loss
        elif self.monitor == 'val_score':
            score = val_score
        print(f'score = {score}')

        # early stopping
        if score is None: return
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
        self.learn.load(self.scoreboard[0]['file'], purge=False)
