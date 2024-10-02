# covnert exposure bracket to HDR output
import argparse 
import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import skimage
import ezexr
from relighting.tonemapper import TonemapHDR


class AttrDict:
    def __init__(self, data):
        self.__dict__['_data'] = data

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data[key] = value

    def __delattr__(self, key):
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __repr__(self):
        return repr(self._data)


def create_default_args():
    args = {}
    args['input_dir'] = ""
    args['output_dir'] = ""
    args['endwith'] = ".png"
    args['ev_string'] = "_ev"
    args['EV'] = "0, -2.5, -5"
    args['gamma'] = 2.4
    args['preview_output'] = False
    return AttrDict(args)


def parse_filename(ev_string, endwith,filename):
    a = filename.split(ev_string)
    name = ev_string.join(a[:-1])
    ev = a[-1].replace(endwith, "")
    ev = int(ev) / 10
    return {
        'name': name,
        'ev': ev,
        'filename': filename
    }


def process_image(args, info):
    
    #output directory
    hdrdir = args.output_dir
    os.makedirs(hdrdir, exist_ok=True)
    
    scaler = np.array([0.212671, 0.715160, 0.072169])
    name = info['name']
    # ev value for each file
    evs = [e for e in sorted(info['ev'], reverse = True)]

    # filename
    files = [info['ev'][e] for e in evs]
    
    # inital first image
    image0 = skimage.io.imread(os.path.join(args.input_dir, files[0]))[...,:3]
    image0 = skimage.img_as_float(image0)
    image0_linear = np.power(image0, args.gamma)

    # read luminace for every image 
    luminances = []
    for i in range(len(evs)):
        # load image
        path = os.path.join(args.input_dir, files[i])
        image = skimage.io.imread(path)[...,:3]
        image = skimage.img_as_float(image)
        
        # apply gama correction
        linear_img = np.power(image, args.gamma)
        
        # convert the brighness
        linear_img *= 1 / (2 ** evs[i])
        
        # compute luminace
        lumi = linear_img @ scaler
        luminances.append(lumi)
        
    # start from darkest image
    out_luminace = luminances[len(evs) - 1]
    for i in range(len(evs) - 1, 0, -1):
        # compute mask
        maxval = 1 / (2 ** evs[i-1])
        p1 = np.clip((luminances[i-1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
        p2 = out_luminace > luminances[i-1]
        mask = (p1 * p2).astype(np.float32)
        out_luminace = luminances[i-1] * (1-mask) + out_luminace * mask
        
    hdr_rgb = image0_linear * (out_luminace / (luminances[0] + 1e-10))[:, :, np.newaxis]
    
    # tone map for visualization    
    hdr2ldr = TonemapHDR(gamma=args.gamma, percentile=99, max_mapping=0.9)
    
    ldr_rgb, _, _ = hdr2ldr(hdr_rgb)
    
    ezexr.imwrite(os.path.join(hdrdir, name+".exr"), hdr_rgb.astype(np.float32))
    if args.preview_output:
        preview_dir = os.path.join(args.output_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        bracket = []
        for s in 2 ** np.linspace(0, evs[-1], 10): #evs[-1] is -5
            lumi = np.clip((s * hdr_rgb) ** (1/args.gamma), 0, 1)
            bracket.append(lumi)
        bracket = np.concatenate(bracket, axis=1)
        skimage.io.imsave(os.path.join(preview_dir, name+".png"), skimage.img_as_ubyte(bracket))
    return None


def convert_ev2hdr(input_dir, output_dir):
    # load arguments
    args = create_default_args()
    args.input_dir = input_dir
    args.output_dir = output_dir
    
    files = sorted(os.listdir(args.input_dir))
    
    #filter file out with file ending
    files = [f for f in files if f.endswith(args.endwith)]
    evs = [float(e.strip()) for e in args.EV.split(",")]
    
    # parse into useful data
    files = [parse_filename(args.ev_string, args.endwith, f) for f in files]
    
    # filter out unused ev
    files = [f for f in files if f['ev'] in evs] 
  
    info = {}
    for f in files:
        if not f['name'] in info:
            info[f['name']] = {}
        info[f['name']][f['ev']] = f['filename']

    infolist = []
    for k in info:
        if len(info[k]) != len(evs):
            print("WARNING: missing ev in ", k)
            continue
        # convert to list data
        infolist.append({'name': k, 'ev': info[k]})
    
    # fn = partial(process_image, args)
    # with Pool(8) as p:
    #     r = list(tqdm(p.imap(fn, infolist), total=len(infolist)))
        
    for i in tqdm(infolist):
        process_image(args, i)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    convert_ev2hdr(args.input_dir, args.output_dir)