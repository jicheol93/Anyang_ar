import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
#from pytorch_metric_learning import miners, losses
import pdb


def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )
    #pdb.set_trace()

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )
    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = torch.sqrt(res)
    return res

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor_Att(torch.nn.Module):
    def __init__(self, nb_classes, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        #self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        #nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, X_att, T):
        P = X_att

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        #neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg/2))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class Proxy_Anchor_NegMargin(torch.nn.Module):
    def __init__(self, nb_classes, hardMargin = True, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)

        self.nb_classes = nb_classes
        self.mrg = mrg
        self.alpha = alpha
        self.hardMargin = hardMargin

    def forward(self, X, X_att, T, M):
        P = X_att

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))

        att_diff_mat = M[T]
        if self.hardMargin:
            neg_margin = torch.where(att_diff_mat < 5, torch.ones_like(att_diff_mat), torch.zeros_like(att_diff_mat))
            neg_margin = neg_margin * 0.05 + self.mrg
        else:
            neg_margin = att_diff_mat * 0.03
            neg_margin += self.mrg

        neg_exp = torch.exp(self.alpha * (cos + neg_margin.cuda()))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class Proxy_NCA_Att(torch.nn.Module):
    def __init__(self, nb_classes, scale):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.scale = scale

    def forward(self, X, X_att, T):
        P = F.normalize(X_att, p = 2, dim = -1)
        X = F.normalize(X, p = 2, dim = -1)
        D = torch.cdist(X, P) ** 2
        #D = my_cdist(X, P) ** 2
        #D = torch.nn.functional.linear(X, P)
        #D = pairwise_distance(torch.cat([X,P]), squared=True)[:X.size()[0], X.size()[0]:]

        T = binarize(T, len(P))
        #T = binarize_and_smooth_labels(T, len(P),0.1)
        loss = torch.sum(-T * F.log_softmax(self.scale*D, -1), -1)
        return loss.mean()
    """
    def forward(self, X, X_att, T):
        loss = 0
        P = X_att
        norm_P = l2_norm(P)
        norm_X = l2_norm(X)
        cos = F.linear(norm_X, norm_P)
        logit = 14 * cos
        loss += F.cross_entropy(logit,T)

        return loss
    """

class Proxy_NCA_Att_semiPos(torch.nn.Module):
    def __init__(self, nb_classes, scale, imb_labels):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.scale = scale
        self.imb_labels = imb_labels

    def forward(self, X, X_att, T, M):
        P = F.normalize(X_att, p = 2, dim = -1)
        X = F.normalize(X, p = 2, dim = -1)
        #D = torch.cdist(X, P) ** 2
        #D = my_cdist(X, P) ** 2
        D = torch.nn.functional.linear(X, P)

        att_diff_mat = M[T]
        T_semi_tmp = torch.where(att_diff_mat == 1, torch.ones_like(att_diff_mat), torch.zeros_like(att_diff_mat))
        T_semi = torch.zeros_like(att_diff_mat)

        for idx, label in enumerate(T.tolist()):
            if label in self.imb_labels:
                T_semi[idx] = T_semi_tmp[idx]
        T = binarize(T, len(P))
        T += T_semi.cuda()
        loss = torch.sum(-T * F.log_softmax(D*self.scale, -1), -1)
        #loss = torch.sum(-T_semi.cuda() * F.log_softmax(D*self.scale, -1), -1)
        return loss.mean()
 


class Reg_Att(torch.nn.Module):
    def __init__(self ):
        torch.nn.Module.__init__(self)
        self.L1loss = torch.nn.L1Loss()
        #self.L1loss = torch.nn.L1Loss(reduction='sum')
        self.MSEloss = torch.nn.MSELoss()
        #self.MSEloss = torch.nn.MSELoss(reduction='sum')

    def forward(self, Att_Feat, Att_Diff):
        p_cosine = torch.nn.functional.linear(l2_norm(Att_Feat), l2_norm(Att_Feat))
        norm_att_diff = -1 * Att_Diff / 7.5 + 1
        #loss = self.L1loss(p_cosine, norm_att_diff.cuda())
        loss = self.MSEloss(p_cosine, norm_att_diff.cuda())
        return loss/2


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
    in_features: size of each input sample
    out_features: size of each output sample
    s: norm of input feature
    m: margin
    cos(theta + m)
    """
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        self.s = s
        self.m = m
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cos, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
        phi = cos * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            phi = torch.where(cos > self.th, phi, cos - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cos.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos) # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class ProxyNCA_ArcFace(torch.nn.Module):
    def __init__(self, nb_classes, sigma = 16, margin = 0.3):
        torch.nn.Module.__init__(self)
        self.sigma = sigma
        self.nb_classes = nb_classes
        self.margin = margin
        self.ArcMarginProduct = ArcMarginProduct(self.sigma, self.margin, easy_margin=False)


    def forward(self, X, X_att, T):
        loss = 0
        P = X_att
        norm_P = l2_norm(P)
        norm_X = l2_norm(X)
        cos = F.linear(norm_X, norm_P)
        logit = self.ArcMarginProduct(cos, T)
        # logit = self.sigma * (cos - self.margin)
        loss += F.cross_entropy(logit,T)

        return loss
    
class Regularizer(nn.Module):
    def __init__(self, nb_classes):
        super(Regularizer, self).__init__()
        self.nb_classes = nb_classes
    def forward(self, att, att_diff_mat):
        mc = self.nb_classes * 1 # for one proxy in one class
        att_diff_mat = att_diff_mat.triu(diagonal=1)
        
        att_1 = att.reshape(1,att.shape[0],-1)
        att_2 = att.reshape(att.shape[0],1,-1)

        att_norm_mat = torch.sum((att_2-att_1)**2, dim=-1).triu(diagonal=1)
        mu = 2.0 / (mc**2 - mc) * torch.sum(att_norm_mat)

        # pdb.set_trace()

        residuals = torch.sum(((att_norm_mat - mu - 2*att_diff_mat.cuda())**2).triu(diagonal=1))
        rw = 2.0 / (mc**2 - mc) * residuals

        return rw 

class Regularizer_cos(nn.Module):
    def __init__(self, nb_classes):
        super(Regularizer_cos, self).__init__()
        self.nb_classes = nb_classes
    def forward(self, att, att_diff_mat):
        mc = self.nb_classes * 1 # for one proxy in one class

        p_cosine = F.linear(l2_norm(att), l2_norm(att))
        #norm_att_diff_mat = -1 * att_diff_mat / 32.5 + 1
        norm_att_diff_mat = 1.0/att_diff_mat
        norm_att_diff_mat = norm_att_diff_mat.triu(diagonal=1).cuda()
 
        att_norm_mat = p_cosine.triu(diagonal=1)
        mu = 2.0 / (mc**2 - mc) * torch.sum(att_norm_mat)                                                                

        residuals = torch.sum(((att_norm_mat - mu - 1*norm_att_diff_mat)**2).triu(diagonal=1))
        rw = 2.0 / (mc**2 - mc) * residuals
                               
        return rw 

class Regularizer_AMD(nn.Module):
    def __init__(self, nb_classes, num_cls_sample):
        super(Regularizer_AMD, self).__init__()
        self.nb_classes = nb_classes
        self.num_cls_sample = num_cls_sample
    def forward(self, att, att_diff_mat):
        mc = self.nb_classes * 1 # for one proxy in one class
        num_cls_sample = torch.FloatTensor(self.num_cls_sample).repeat(len(self.num_cls_sample),1).triu(diagonal=1)
        
        att_1 = att.reshape(1,att.shape[0],-1)
        att_2 = att.reshape(att.shape[0],1,-1)

        att_norm_mat = torch.sum((att_2-att_1)**2, dim=-1).triu(diagonal=1)
        mu = 2.0 / (mc**2 - mc) * torch.sum(att_norm_mat)

        # pdb.set_trace()

        residuals = torch.sum(((att_norm_mat - mu - 2*num_cls_sample.cuda())**2).triu(diagonal=1))
        rw = 2.0 / (mc**2 - mc) * residuals
        #rw = 1.0 / mc * residuals

        return rw 

"""
# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
"""
