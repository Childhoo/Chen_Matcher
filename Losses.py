import torch
import torch.nn as nn
import sys

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p


def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss
def loss_HardNegC(anchor, positive, margin = 1.0):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    #tensor.detach() set the requirements of gradients as false
    dist_matrix_detach = distance_matrix_vector(anchor, positive.detach()) + eps
    pos1 = distance_vectors_pairwise(anchor,positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix_detach.size(1))).cuda()
    # steps to filter out same patches that occur in distance matrix as negatives
    dist_without_min_on_diag = dist_matrix_detach + eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    loss = torch.clamp(margin + pos1 - min_neg, min=0.0)
    loss = 0.5 * torch.mean(loss)
    dist_matrix_detach2 = distance_matrix_vector(anchor.detach(), positive) + eps
    # steps to filter out same patches that occur in distance matrix as negatives
    dist_without_min_on_diag2 = dist_matrix_detach2 + eye*10
    mask2 = (dist_without_min_on_diag2.ge(0.008).float()-1)*-1
    mask2 = mask2.type_as(dist_without_min_on_diag2)*10
    dist_without_min_on_diag2 = dist_without_min_on_diag2 + mask2
    min_neg2 = torch.min(dist_without_min_on_diag2,0)[0]
    loss += 0.5 * torch.clamp(margin + pos1 - min_neg2, min=0.0).mean()
    return loss

def loss_L2Net(anchor, positive, anchor_swap = False,  margin = 1.0, loss_type = "triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1);
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix),1) + eps;
        loss = -torch.log( exp_pos / exp_den )
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix),0) + eps;
            loss += -torch.log( exp_pos / exp_den1 )
    else: 
        print ('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #output = cos(input1, input2)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet_Chen(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        print(loss.shape)
        print(pos.shape)
        print(min_neg.shape)
#        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) #this is not desc to desc loss
#        cos_dis = cos(pos, min_neg)
#        loss = loss + 0.1*cos_dis.abs_()
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
#        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
        loss = torch.clamp(margin - min_neg, min=0.0) + torch.clamp(pos/(min_neg+1e-8), min=1.0, max=50.0)*pos
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet_k_hardest(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min_hardest_k', k=3, loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and k closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*100
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min_hardest_k':
        min_k_neg = (-1.0)*(0-dist_without_min_on_diag).topk(k,1)[0]
        min_neg = min_k_neg.mean(1)
        pos = pos1
    elif batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #output = cos(input1, input2)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def loss_HardNet_k_hardest_4_branches(anchor, positive, anchor_aux, positive_aux, \
                                      anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min_hardest_k', k=3, loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and k closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix_a_p = distance_matrix_vector(anchor, positive) +eps
    dist_matrix_a_paux = distance_matrix_vector(anchor, positive_aux) +eps
    dist_matrix_aaux_p = distance_matrix_vector(anchor_aux, positive) +eps
    dist_matrix_aaux_paux = distance_matrix_vector(anchor_aux, positive_aux) +eps
    
    dist_matrix_all = torch.cat((dist_matrix_a_p.unsqueeze(2),dist_matrix_a_paux.unsqueeze(2), \
                             dist_matrix_aaux_p.unsqueeze(2),dist_matrix_aaux_paux.unsqueeze(2)), dim=2)
    
    dist_matrix = torch.min(dist_matrix_all, 2)[0] #use the minimum in each position as the negative distance
    
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
#    pos1 = torch.diag(dist_matrix)
    #distance for positive, pick the largest one in each position
    dist_matrix_4pos = torch.max(dist_matrix_all, 2)[0]
    pos1 = torch.diag(dist_matrix_4pos)
    
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*100
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    
    
    if batch_reduce == 'min_hardest_k':
        min_k_neg = (-1.0)*(0-dist_without_min_on_diag).topk(k,1)[0]
        min_neg = min_k_neg.mean(1)
        pos = pos1
    elif batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #output = cos(input1, input2)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

#the hardnet loss, in which a weak match is contained
def loss_HardNet_WeakMatch(anchor, positive, positive_weak, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', k=3, loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and k closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix_a_p = distance_matrix_vector(anchor, positive) +eps
    dist_matrix_a_pweak = distance_matrix_vector(anchor, positive_weak) +eps
    dist_matrix_p_pweak = distance_matrix_vector(positive, positive_weak) +eps
    
    #derive the minimum distance among d(a,p_w), d(a,p), d(p,p_w)    
    dist_matrix_all = torch.cat((dist_matrix_a_p.unsqueeze(2),dist_matrix_a_pweak.unsqueeze(2), \
                             dist_matrix_p_pweak.unsqueeze(2)), dim=2)
    dist_matrix_allpos = torch.max(dist_matrix_all, 2)[0]
    pos1 = torch.diag(dist_matrix_allpos) #find the maximum positive distance for three possible combinations
    
    
    dist_matrix = torch.min(dist_matrix_all, 2)[0] #use the minimum in each position as the negative distance
    
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives

    
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*100
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    
    
    if batch_reduce == 'min_hardest_k':
        min_k_neg = (-1.0)*(0-dist_without_min_on_diag).topk(k,1)[0]
        min_neg = min_k_neg.mean(1)
        pos = pos1
    elif batch_reduce == 'min':
        #always use the closest pair as unmatched pair
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        # do anchor swap all the time
        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        min_neg = torch.min(min_neg,min_neg2)
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
#        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #output = cos(input1, input2)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss