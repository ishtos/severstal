import numpy as np

def run_length_encode(component):
    if component.sum() == 0:
        return ''
    component = np.hstack([np.array([0]), component.T.flatten(), np.array([0])])
    start  = np.where(component[1: ] > component[:-1])[0]
    end    = np.where(component[:-1] > component[1: ])[0]
    length = end-start
      
    rle = []
    for i in range(len(length)):
        if i==0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i], length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle

def run_length_decode(rle, height=256, width=1600, fill_value=1.):
    component = np.zeros((height, width), np.float32)
    if str(rle) == 'nan' or rle == 0:
        return component
    component = component.reshape(-1)
    rle  = np.array([int(s) for s in rle.split(' ')])
    rle  = rle.reshape(-1, 2)
    start = 0
    for index,length in rle:
        start = index
        end   = start+length
        component[start : end] = fill_value
        start = end

    component = component.reshape(width, height).T
    return component

def build_mask(s, height, width):
    mask = np.zeros((height, width, 4))
    for i in range(4):
        mask[:,:,i] = run_length_decode(s[f'{i+1}'], height, width)
    return mask
