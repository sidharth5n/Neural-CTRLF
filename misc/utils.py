import numpy as np
import os
from skimage.filters import threshold_otsu
from queue import Queue
from threading import Thread, Lock
import cv2
try:
    import cPickle as pickle
except:
    import pickle
import torch

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_checkpoint(path, x = None, device = None):
    if os.path.exists(path):
        if x:
            x.load_state_dict(torch.load(path, map_location=device))
        else:
            return load(path)
    else:
        raise Exception("Checkpoint does not exist. Cannot resume training.")

def filter_region_proposals(region_proposals, scale):
    """
    Remove duplicate region proposals when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.

    Parameters
    ----------
    data             : list
    original_heights : numpy.ndarray
    original_widths  : numpy.ndarray
    image_size       : int
    """
    if isinstance(region_proposals, list):
        region_proposals = np.array(region_proposals, dtype = np.int32)
    # Since we downsample the image 8 times before the roi-pooling, 
    # divide scaling by 8. Hardcoded per network architecture.
    scale /= 8
    xy, wh = np.split(region_proposals, 2, -1)
    # Scale x and y coordinates
    xy = np.minimum(np.round(scale * (xy) + 1).astype(int), 1)
    # Scale width and height
    wh = np.round(scale * wh).astype(int)
    # Find indices of boxes with positive width and height
    indices = np.prod(wh, -1) > 0
    # Only keep unique proposals in downsampled coordinate system, i.e., remove aliases 
    region_proposals, _ = unique_boxes(region_proposals[indices])
    return region_proposals

def unique_boxes(boxes):
    tmp = np.array(boxes)
    ncols = tmp.shape[1]
    dtype = tmp.dtype.descr * ncols
    struct = tmp.view(dtype)
    uniq, index = np.unique(struct, return_index=True)
    tmp = uniq.view(tmp.dtype).reshape(-1, ncols)
    return tmp, index

def close_crop_box(img, box):
    gray = img[box[1]:box[3], box[0]:box[2]]
    t_img = gray < threshold_otsu(gray)
    v_proj = t_img.sum(axis=1)
    h_proj = t_img.sum(axis=0)
    y1o = box[1] + max(v_proj.nonzero()[0][0] - 1, 0)
    x1o = box[0] + max(h_proj.nonzero()[0][0] - 1, 0)
    y2o = box[3] - max(v_proj.shape[0] - v_proj.nonzero()[0].max() - 1, 0)
    x2o = box[2] - max(h_proj.shape[0] - h_proj.nonzero()[0].max() - 1, 0)
    obox = (x1o, y1o, x2o, y2o)
    return obox

def augment(word, tparams, keep_size = False):
    """
    Applies an affine transformation of the image followed by erosion
    or dilation.
    """
    assert(word.ndim == 2)
    t = np.zeros_like(word)
    s = np.array(word.shape) - 1
    # Copy boundary contents
    t[0, :] = word[0, :]
    t[:, 0] = word[:, 0]
    t[s[0], :] = word[s[0], :]
    t[:, s[1]] = word[:, s[1]]
    # Find median value of the boundary pixels
    pad = np.median(t[t > 0])
    # Pad the input image 4 pixels wide on each side with the median value
    out = cv2.copyMakeBorder(word, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, pad)
    # Perform an affine transformation on the padded image
    out = affine(out, tparams)
    # Tight crop the transformed image
    out = aug_crop(out, tparams) 
    # Perform one of erosion or dilation       
    out = morph(out, tparams)
    # Resize the image to input size
    if keep_size:
        out = tf.resize(out, word.shape)
    out = np.round(out).astype(np.ubyte)
    return out

def affine(img, tparams):
    phi = (np.random.uniform(tparams['shear'][0], tparams['shear'][1])/180) * np.pi
    theta = (np.random.uniform(tparams['rotate'][0], tparams['rotate'][1])/180) * np.pi
    t = tf.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = tf.warp(img, t, order=tparams['order'], mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp

def aug_crop(img, tparams):
    t_img = img < threshold_otsu(img)
    nz = t_img.nonzero()
    pad = np.random.randint(low = tparams['hpad'][0], high = tparams['hpad'][1], size=2)    
    vpad = np.random.randint(low = tparams['vpad'][0], high = tparams['vpad'][1], size=2)    
    b = [max(nz[1].min() - pad[0], 0), max(nz[0].min() - vpad[0], 0), 
         min(nz[1].max() + pad[1], img.shape[1]), min(nz[0].max() + vpad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]

def morph(img, tparams):
    ops = [mor.grey.erosion, mor.grey.dilation]
    t = ops[np.random.randint(2)] 
    if t == 0:    
        selem = mor.square(np.random.randint(1, tparams['selem_size'][0]))
    else:
        selem = mor.square(np.random.randint(1, tparams['selem_size'][1]))
    return t(img, selem)

def create_background(m, shape, fstd = 2, bstd = 10):
    canvas = np.ones(shape) * m
    noise = np.random.randn(*shape) * bstd
    noise = cv2.GaussianBlur(noise, ksize = (0,0), sigmaX = fstd)     #low-pass filter noise
    canvas += noise
    canvas = np.round(canvas).astype(np.uint8)
    return canvas

def build_vocab(data):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    texts = []
    for datum in data:
        if datum['split'] == 'train':
            for r in datum['regions']:
                texts.append(r['label'])
    vocab, indices = np.unique(texts, return_index=True)
    return vocab, indices

def build_vocab_dict(vocab):
    wtoi = {w : i for i, w in enumerate(vocab, start = 1)}
    wtoi['unk'] = 0
    return wtoi

def build_filenames(data):
    filenames = []
    for datum in data:
        _, fname = os.path.split(datum['id'])
        fname, _ = os.path.splitext(fname)
        filenames.append(fname)
    return filenames

def encode_word_embeddings(datum, wtoe):
    """
    Encode each label as a word embedding
    """
    we = []
    # for datum in data:
    for r in datum['regions']:
        we.append(wtoe.get(r['label'], wtoe['unk']))
            
    return np.array(we)
    
def encode_labels(datum, wtoi):
    """
    Encode each label as an integer
    """
    labels = []
    # for datum in data:
    for r in datum['regions']:
        labels.append(wtoi.get(r['label'], wtoi['unk']))      
    return np.array(labels)

def pad_proposals(proposals, im_shape, pad=10):
    xy, wh = np.split(proposals, 2, -1)
    xy = np.maximum(xy - pad, 0)
    wh = np.minimum(wh + pad, im_shape)
    proposals = np.concatenate([xy, wh], -1)
    return proposals

def convert_boxes(boxes, scale, max_image_size, box_type='gt_boxes'):
    """
    Scales boxes according the gives dimensions and converts boxes to (xc, yc, w, h)
    format.

    Parameters
    ----------
    boxes          : numpy.ndarray or list of shape (P, 4)
                     Coordinates of the bboxes in (x, y, w, h) format
    scale          : float
                     Scaling factor with which the image was scaled
    max_image_size : tuple
                     Size of the largest image in the dataset in (W, H) format
    box_type       : str
                     One of gt_boxes, region_proposals

    Returns
    -------
    boxes          : numpy.ndarray of shape (P, 4)
                     Scaled bboxes in (xc, yc, w, h) format
    """
    if isinstance(boxes, list):
        boxes = np.array(boxes)       
    # Separate xy and wh for easier computation
    xy, wh = np.split(boxes, 2, -1)
    # Scale x and y and clip to [1, image_size - 1]
    xy = np.clip(np.round(scale * (xy - 1) + 1).astype(int), 1, max_image_size - 1)
    # Scale w and h and clip to [1, image_size - x/y]
    wh = np.clip(np.round(scale * wh).astype(int) + xy, xy + 1, max_image_size - 1) - xy
    # Convert to xc, yc, w, h format
    xcyc = xy + np.floor(wh / 2)
    # Concatenate the boxes
    boxes = np.concatenate([xcyc, wh], -1).astype(int)
    return boxes

def build_img_idx_to_box_idxs(data, boxes='regions'):
    img_idx = 0
    box_idx = 0
    num_images = len(data)
    img_to_first_box = np.zeros(num_images, dtype=np.int32)
    img_to_last_box = np.zeros(num_images, dtype=np.int32)
    for datum in data:
        img_to_first_box[img_idx] = box_idx
        box_idx += len(datum[boxes])
        img_to_last_box[img_idx] = box_idx - 1 # -1 to make these inclusive limits
        img_idx += 1
  
    return img_to_first_box, img_to_last_box

def custom_function(root, outdir, data, image_mean, image_size, max_image_size, wtoi, wtoe, num_workers=5):
        
    lock = Lock()
    q = Queue()

    for i, datum in enumerate(data):
        q.put((i, os.path.join(root, 'gw_20p_wannot', datum['file']), data[i]))
    
    def worker():
        while True:
            # Get data from the queue
            i, filename, datum = q.get()
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # Get original height and width of the image
            H0, W0 = img.shape[0], img.shape[1]
            # Find scale such that larger side is of length image_size
            scale = float(image_size) / max(H0, W0)
            # Resize image such that larger side is of length image_size
            img = cv2.resize(img, dsize = None, fx = scale, fy = scale)
            # Get new height and width of the image
            H, W = img.shape
            # Invert the image and add a channel dimension
            img = np.invert(img)[np.newaxis]
            # Encode dct embeddings
            dct_word_embeddings = encode_word_embeddings(datum, wtoe)
            # Rescale boxes
            gt_boxes = convert_boxes(datum['gt_boxes'], scale, max_image_size)
            # Encode labels
            labels = encode_labels(datum, wtoi)
            # Keep proposals which are valid and unique in the downsampled space
            region_proposals = filter_region_proposals(datum['region_proposals'], scale)
            # Update datum with the new proposals
            datum['region_proposals'] = region_proposals.tolist()
            # Pad proposals - this is required for washington dataset
            region_proposals = pad_proposals(region_proposals, [H, W], 10)
            # Rescale region proposals
            region_proposals = convert_boxes(region_proposals, scale, max_image_size)
            lock.acquire()
            if i % 10 == 0:
                print(f'Processed {i}/{len(data)}')
            
            np.savez_compressed(os.path.join(outdir, datum['split'], datum['file'].replace('.tif', '.npz')),
                                image = np.array((img - image_mean) / 255, dtype = np.float32),
                                embeddings = np.array(dct_word_embeddings, dtype = np.float32),
                                labels = np.array(labels, dtype = np.int64),
                                region_proposals = region_proposals,
                                boxes = gt_boxes,
                                infos = np.array([H0, W0, H, W], dtype = np.int32))

            lock.release()
            q.task_done()
      
    print('Rescaling images and boxes, encoding labels and embedding, filtering proposals!')
    for i in range(num_workers):
        t = Thread(target = worker)
        t.daemon = True
        t.start()
    q.join()

def replace_tokens(text, tokens):
    """
    Remove all occurrences of each token in the given text.
    
    Parameters
    ----------
    text   : str
    tokens : list

    Returns
    -------
    text   : str
    """
    for t in tokens:
        text = text.replace(t, '')
    return text #str(text, errors = 'replace')