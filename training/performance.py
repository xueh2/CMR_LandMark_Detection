
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

"""
These utility functions are duplicated mainly from 
https://github.com/ternaus/robot-surgery-segmentation

for the implementation of jaccard index

"""

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def check_accuracy_test(loader, model, criterion, device=torch.device('cpu'), dtype=torch.float): 
    num_correct = 0
    num_samples = 0
    running_loss = 0.0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            if device is None:
                x = x.to(dtype).cuda() 
                y = y.to(torch.long).cuda()
            else:
                x = x.to(device=device, dtype=dtype) 
                y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = criterion(scores, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            running_loss += loss.item() * x.shape[0]
        acc = float(num_correct) / num_samples
        loss = running_loss/ num_samples
        # print('\033[1m' + 'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc, loss

def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, entropy_weight=1.0, jaccard_weight=0, debug_mode=False):
        self.nll_loss = nn.BCELoss()
        self.entropy_weight = entropy_weight
        self.jaccard_weight = jaccard_weight        
        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets):
        
        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        probs = torch.sigmoid(outputs)

        if(self.entropy_weight>0):                    
            loss = self.entropy_weight * self.nll_loss(probs, x)
        else:
            loss = 0.0

        if(self.debug_mode):     
            print('Binary cross-entropy loss is ', loss)
            
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (x == 1).float()
            jaccard_output = probs

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            jaccard_loss = self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
            if(self.debug_mode):     
                print('jaccard loss is ', -jaccard_loss)
            
            loss -= jaccard_loss
        return loss

class LossMultiOld:
    """
    This loss function is for multi-class segmentation with ground-truth to be one-hot mask.
    This loss is a weighted sum of the KLDivLoss and jaccard loss
    targets: [N, ...], one-host mask of target classes for every pixel
    outputs: [N,C, ...], logits
    """
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        
        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        log_prob = torch.log_softmax(outputs, dim=1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(log_prob, x)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (x == cls).float()
                jaccard_output = log_prob[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

class LossMulti:
    """
    This loss function is for multi-class segmentation with ground-truth to be one-hot mask.
    This loss is a weighted sum of the KLDivLoss and jaccard loss
    targets: [N, ...], one-host mask of target classes for every pixel
    outputs: [N,C, ...], logits
    """
    def __init__(self, entropy_weight=1.0, jaccard_weight=0, class_weights=None, debug_mode=False):
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None

        self.nll_loss = nn.NLLLoss(weight=self.class_weights)
        self.entropy_weight = entropy_weight
        self.jaccard_weight = jaccard_weight
        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets):
        
        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        #self.num_classes = outputs.shape[1]
        self.num_classes = 1
        
        log_prob = torch.log_softmax(outputs, dim=1)

        if(self.entropy_weight>0):
            loss = self.entropy_weight * self.nll_loss(log_prob, x)
        else:
            loss = 0.0

        if(self.debug_mode):     
            print('cross-entropy loss is ', loss)
        
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    jaccard_target = (x == cls).float()
                    jaccard_output = log_prob[:, cls].exp()

                    intersection = (jaccard_output * jaccard_target).sum()
                    union = jaccard_output.sum() + jaccard_target.sum()

                    cls_loss = torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)
                        
                    if(self.class_weights is not None):
                        jaccard_loss -= cls_loss * self.class_weights[cls]
                    else:
                        jaccard_loss -= cls_loss
            
            if(self.debug_mode):     
                print('log jaccard_loss is ', jaccard_loss)
                
            loss += jaccard_loss
            
        return loss

class LossMultiSoftProb:
    """
    This loss function is for multi-class segmentation with ground-truth to be soft probablity.
    This loss is a weighted sum of the soft cross entropy and jaccard loss
    targets: [N,C, ...], probablities of target for every pixel
    outputs: [N,C, ...], logits
    """
    def __init__(self, entropy_weight=1.0, jaccard_weight=0, class_weights=None, debug_mode=False):
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None

        self.entropy_weight = entropy_weight
        self.jaccard_weight = jaccard_weight        
        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets):

        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        self.num_classes = x.shape[1]
        
        log_prob = torch.log_softmax(outputs, dim=1)

        # compute soft cross-entropy
        if(self.entropy_weight>0):
            loss = -torch.mean( torch.sum(log_prob * x, dim=1) ) * self.entropy_weight
        else:
            loss = 0.0

        if(self.debug_mode):     
            print('soft cross-entropy loss is ', loss)
                
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    jaccard_target = x[:, cls].float()
                    jaccard_output = log_prob[:, cls].exp()

                    intersection = (jaccard_output * jaccard_target).sum()
                    union = jaccard_output.sum() + jaccard_target.sum()

                    cls_loss = torch.log((intersection) / (union - intersection + eps)) * self.jaccard_weight
                    
                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)
                        
                    if(self.class_weights is not None):
                        jaccard_loss -= cls_loss * self.class_weights[cls]
                    else:
                        jaccard_loss -= cls_loss
                    
            if(self.debug_mode):     
                print('log jaccard_loss is ', jaccard_loss)
                
            loss += jaccard_loss
            
        return loss

class LossMultiSoftProb_KLD_Jaccard:
    """
    This loss function is for multi-class segmentation with ground-truth to be probablities, rather than one-hot masks.
    This loss is a weighted sum of the KLDivLoss and jaccard loss
    targets: [N,C, ...], probablity of target classes for every pixel
    outputs: [N,C, ...], logits
    """
    def __init__(self, KLD_weight=1.0, jaccard_weight=0, class_weights=None, reduction='batchmean', debug_mode=False):
        
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None
            
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.KLD_weight = KLD_weight
        self.jaccard_weight = jaccard_weight        

        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets):
        
        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        self.num_classes = x.shape[1]
        
        log_prob = torch.log_softmax(outputs, dim=1)

        if(self.KLD_weight>0):
            loss = self.kldiv_loss(log_prob, x) * self.KLD_weight
        else:
            loss = 0.0

        if(self.debug_mode):
            print('KL D loss is ', loss)
        
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    jaccard_target = x[:, cls].float()
                    jaccard_output = log_prob[:, cls].exp()

                    intersection = (jaccard_output * jaccard_target).sum()
                    union = jaccard_output.sum() + jaccard_target.sum()

                    cls_loss = torch.log((intersection) / (union - intersection + eps)) * self.jaccard_weight
                    
                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)
                        
                    if(self.class_weights is not None):
                        jaccard_loss -= cls_loss * self.class_weights[cls]
                    else:
                        jaccard_loss -= cls_loss
                    
            if(self.debug_mode):     
                print('log jaccard_loss is ', jaccard_loss)
                
            loss += jaccard_loss
            
        return loss

class LossMultiSoftProb_KLD_Dice:
    """
    This loss function is for multi-class segmentation with ground-truth to be probablities, rather than one-hot masks.
    This loss is a weighted sum of the KLDivLoss and dice loss
    targets: [N,C, ...], probablity of target classes for every pixel
    outputs: [N,C, ...], logits
    """
    def __init__(self, KLD_weight=1.0, dice_weight=1.0, class_weights=None, reduction='batchmean', debug_mode=False):
        
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None
                    
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.dice_weight = dice_weight        
        self.KLD_weight = KLD_weight

        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets):
        
        if(type(targets)==list):
            x = targets[0]
        else:
            x = targets

        self.num_classes = x.shape[1]
                
        log_prob = torch.log_softmax(outputs, dim=1)
        
        if(self.KLD_weight>0):
            loss = self.kldiv_loss(log_prob, x) * self.KLD_weight
        else:
            loss = 0.0

        if(self.debug_mode):
            print('KL D loss is ', loss)
        
        if self.dice_weight>0:
            eps = 1e-15
            dice_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    target = x[:, cls].float()
                    output = log_prob[:, cls].exp()

                    # for every class, compute dice for every sample
                    numerator = 2. * torch.sum(output * target, dim=(1, 2))
                    denominator = torch.sum(torch.square(output) + torch.square(target), dim=(1, 2))

                    cls_loss = torch.log( torch.mean(numerator / (denominator + eps)) )
                    cls_loss *= self.dice_weight

                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)

                    if(self.class_weights is not None):
                        dice_loss -= cls_loss * self.class_weights[cls]
                    else:
                        dice_loss -= cls_loss
                    
            if(self.debug_mode):     
                print('log dice_loss is ', dice_loss)
                
            loss += dice_loss
            
        return loss

class LossMultiSoftProb_KLD_Dice_L2Dist:
    """
    This loss function is for multi-task learning for mult-class segmentation and L2 distance.
    This loss is a weighted sum of the KLDivLoss and dice loss
    (targets, pos): [N,C, ...], probablity of target classes for every pixel and pos [N, 2], locations of landmarks
    (outputs, pot): [N,C, ...], logits and estimated locaitons
    """
    def __init__(self, KLD_weight=1.0, dice_weight=1.0, class_weights=None, reduction='batchmean', l2_dist_weight=0.0, debug_mode=False):
        
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None
                    
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.dice_weight = dice_weight        
        self.KLD_weight = KLD_weight
        self.l2_dist_weight = l2_dist_weight

        self.debug_mode = debug_mode
        
    def __call__(self, outputs_all, targets_all):
        
        outputs, out_pos = outputs_all
        targets = targets_all[0]
        pos = targets_all[1]

        self.num_classes = targets.shape[1]
                
        log_prob = torch.log_softmax(outputs, dim=1)
        
        if(self.KLD_weight>0):
            loss = self.kldiv_loss(log_prob, targets) * self.KLD_weight
        else:
            loss = 0.0

        if(self.debug_mode):
            print('KL D loss is ', loss)
        
        if self.dice_weight>0:
            eps = 1e-15
            dice_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    target = targets[:, cls].float()
                    output = log_prob[:, cls].exp()

                    # for every class, compute dice for every sample
                    numerator = 2. * torch.sum(output * target, dim=(1, 2))
                    denominator = torch.sum(torch.square(output) + torch.square(target), dim=(1, 2))

                    cls_loss = torch.log( torch.mean(numerator / (denominator + eps)) )
                    cls_loss *= self.dice_weight

                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)

                    if(self.class_weights is not None):
                        dice_loss -= cls_loss * self.class_weights[cls]
                    else:
                        dice_loss -= cls_loss
                    
            if(self.debug_mode):     
                print('log dice_loss is ', dice_loss)
                
            loss += dice_loss

        if(self.l2_dist_weight>0):
            l2_loss = nn.MSELoss()
            v = l2_loss(pos, out_pos) * self.l2_dist_weight
            if(self.debug_mode):     
                print('l2 dist loss is ', v)

            loss += v

        return loss

class LossMultiSoftProb_KLD_Dice_L2DistMap:
    """
    This loss function is for multi-task learning for mult-class segmentation and L2 distance map.
    This loss is a weighted sum of the KLDivLoss and dice loss
    targets: [N,C, ...], probablity of target classes for every pixel
    (outputs, pot): [N,C, ...], logits and estimated locaitons
    """
    def __init__(self, KLD_weight=1.0, dice_weight=1.0, class_weights=None, reduction='batchmean', l2_dist_weight=0.0, debug_mode=False):
        
        if class_weights is not None:
            self.class_weights = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None
                    
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.dice_weight = dice_weight        
        self.KLD_weight = KLD_weight
        self.l2_dist_weight = l2_dist_weight

        self.debug_mode = debug_mode
        
    def __call__(self, outputs_all, targets_all):
        
        outputs = outputs_all
        targets = targets_all[0]
        pos = targets_all[1]
        dm = targets_all[2]

        self.num_classes = targets.shape[1]
                
        log_prob = torch.log_softmax(outputs, dim=1)
        
        if(self.KLD_weight>0):
            loss = self.kldiv_loss(log_prob, targets) * self.KLD_weight
        else:
            loss = 0.0

        if(self.debug_mode):
            print('KL D loss is ', loss)
        
        if self.dice_weight>0:
            eps = 1e-15
            dice_loss = 0
            for cls in range(self.num_classes):
                if(self.class_weights[cls]>0):
                    target = targets[:, cls].float()
                    output = log_prob[:, cls].exp()

                    # for every class, compute dice for every sample
                    numerator = 2. * torch.sum(output * target, dim=(1, 2))
                    denominator = torch.sum(torch.square(output) + torch.square(target), dim=(1, 2))

                    cls_loss = torch.log( torch.mean(numerator / (denominator + eps)) )
                    cls_loss *= self.dice_weight

                    if(self.debug_mode):
                        print('   class loss is ', -cls_loss)

                    if(self.class_weights is not None):
                        dice_loss -= cls_loss * self.class_weights[cls]
                    else:
                        dice_loss -= cls_loss
                    
            if(self.debug_mode):     
                print('log dice_loss is ', dice_loss)
                
            loss += dice_loss

        if(self.l2_dist_weight>0):
            v = 0
            for cls in range(1, self.num_classes):
                if(self.class_weights[cls]>0):
                    output = log_prob[:, cls].exp()
                    dm_cls = dm[:,cls-1].float()

                    #print(dm_cls.shape)
                    sv = torch.sum(output * dm_cls, dim=(1,2))
                    #print(sv, torch.mean(sv))
                    v -= torch.log(torch.mean(sv)) * self.class_weights[cls]

                    if(self.debug_mode):     
                        print('l2 dist loss is ', v)

            loss += v * self.l2_dist_weight

        return loss

class Loss_L2Dist:
    """
    This loss function is for landmark detection using L2 distance.
    (targets, pos): [N,C, ...], probablity of target classes for every pixel and pos [N, 2], locations of landmarks
    outputs : [N,m], estimated locaitons
    """
    def __init__(self, reduction="mean", debug_mode=False):
                            
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.debug_mode = debug_mode
        
    def __call__(self, outputs, targets_all):
        
        targets = targets_all[0]
        pos = targets_all[1]
        loss = self.mse_loss(pos, outputs)
        if(self.debug_mode):    
            print("outputs is ", outputs.shape) 
            print("pos is ", pos.shape) 
            print('l2 dist loss is ', loss)

        return loss

def create_loss_func(name='LossMulti', class_weights=None, entropy_weight=1.0, overlap_weight=1.0, dist_weight=0.0, debug_mode=False):
    '''
    create loss function
    '''
    
    if(name=='LossMultiSoftProb_KLD_Dice_L2Dist'):
        loss = LossMultiSoftProb_KLD_Dice_L2Dist(KLD_weight=entropy_weight, 
                                        dice_weight=overlap_weight, 
                                        class_weights=class_weights, 
                                        reduction='batchmean', 
                                        l2_dist_weight=dist_weight,
                                        debug_mode=debug_mode)
        return loss, name

    if(name=='LossMultiSoftProb_KLD_Dice'):
        loss = LossMultiSoftProb_KLD_Dice(KLD_weight=entropy_weight, 
                                        dice_weight=overlap_weight, 
                                        class_weights=class_weights, 
                                        reduction='batchmean', 
                                        debug_mode=debug_mode)
        return loss, name

    if(name=='LossMultiSoftProb_KLD_Jaccard'):
        loss = LossMultiSoftProb_KLD_Jaccard(KLD_weight=entropy_weight, 
                                            jaccard_weight=overlap_weight, 
                                            class_weights=class_weights, 
                                            reduction='batchmean', 
                                            debug_mode=debug_mode)
        return loss, name

    if(name=='LossMultiSoftProb'):
        loss = LossMultiSoftProb(entropy_weight=entropy_weight, 
                                jaccard_weight=overlap_weight, 
                                class_weights=class_weights, 
                                debug_mode=debug_mode)
        return loss, name

    if(name=='LossMulti'):
        loss = LossMulti(entropy_weight=entropy_weight, 
                        jaccard_weight=overlap_weight, 
                        class_weights=class_weights, 
                        debug_mode=debug_mode)
        return loss, name

    if(name=='LossBinary'):    
        loss = LossBinary(entropy_weight=entropy_weight, 
                        jaccard_weight=overlap_weight, 
                        debug_mode=debug_mode)
        return loss, name

    raise "Unknown loss name"