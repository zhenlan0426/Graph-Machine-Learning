# teach myself geometric deep learning. resources I find helpful 
https://github.com/gordicaleksa/pytorch-GAT/

https://github.com/AntonioLonga/PytorchGeometricTutorial/

def min_k(piles,H):
    def round_up(n,k):
        return n // k + (n % k > 0)
    def evaluate(piles,k):
        return sum([round_up(p,k) for p in piles])
    left,right = 1,-1
    sum_ = 0
    for p in piles:
        if p > right:
            right = p
        sum_ += p
    if sum_ <= H: return 1
    index = -1
    while left < right:
        diff = right - left
        if diff == 1:
            if index == -1:
                return right
            else:
                return index
        mid = left + diff//2
        mid_v = evaluate(piles,mid)
        if mid_v == H:
            index = mid
            right = mid
        elif mid_v < H:
            right = mid
        else:
            left = mid

def min_k(weights,days):
    def round_up(n,k):
        return n // k + (n % k > 0)

    def evaluate(weights,k,n):
        sum_ = 0
        count = 0
        for i in range(n):
            sum_ += weights[i]
            if sum_ > k:
                count += 1
                sum_ = weights[i]
        return count + round_up(sum_,k)
    
    left,right,n = 0,0,0
    for w in weights:
        right += w
        n += 1
        if w > left:
            left = w
    
    if evaluate(weights,left,n) < days: return left
    index = -1
    while left < right:
        diff = right - left
        if diff == 1:
            if index == -1:
                return right
            else:
                return index
        mid = left + diff//2
        mid_v = evaluate(weights,mid,n)
        if mid_v == days:
            index = mid
            right = mid
        elif mid_v < days:
            right = mid
        else:
            left = mid        
