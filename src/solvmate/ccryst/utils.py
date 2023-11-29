import math
import random
import re
import numpy as np
from typing import List

import PIL
from PIL import ImageDraw
from PIL.Image import Image
from joblib import Parallel,delayed

import pandas as pd

from collections import defaultdict

from contextlib import redirect_stderr,redirect_stdout
import io

def safe(fun):
    """
    Wraps a function so that exceptions are
    not raised but yield a None instead.
    """
    def w(*args,**kwargs):
        try:
            return fun(*args,**kwargs)
        except:
            return None
    return w

class Silencer:
    """
    A useful tool for silencing stdout and stderr.
    Usage:
    >>> with Silencer() as s:
    ...         print("kasldjf")

    >>> print("I catched:",s.out.getvalue())
    I catched: kasldjf
    <BLANKLINE>

    Note that nothing was printed and that we can later
    access the stdout via the out field. Similarly,
    stderr will be redirected to the err field.
    """
    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()
    
    def __enter__(self):
        self.rs = redirect_stdout(self.out)
        self.re = redirect_stderr(self.err)
        self.rs.__enter__()
        self.re.__enter__()
        return self
    
    def __exit__(self, exctype, excinst, exctb):
        self.rs.__exit__(exctype,excinst,exctb)
        self.re.__exit__(exctype,excinst,exctb)

def alert_jupyter():
    """
    Makes a beep sound to inform the user about the finished result.
    Very useful for long running jupyter notebooks!
    """
    from IPython.display import display,HTML
    display(HTML(
        """
        <script>
            var context = new (window.AudioContext || window.webkitAudioContext)();
            var osc = context.createOscillator(); // instantiate an oscillator
            osc.type = 'sine'; // this is the default - also square, sawtooth, triangle
            osc.frequency.value = 440+loop*50; // Hz
            osc.connect(context.destination); // connect it to the destination
            osc.start(); // start the oscillator
            osc.stop(context.currentTime + 1); // stop 2 seconds after the current time
            
        </script>
        """
        ))

def flatten_once(lst):
    """
    Flattens the given datastructure one level.
    For a list this means removing nesting by
    concatenating listed lists in a single list:
    >>> flatten_once([["a"],["b","c"]])
    ['a', 'b', 'c']
    """
    return sum(lst,[])

def in_chunks(inputs,num_chunks:int):
    """
    Splits the given inputs into chunks

    >>> in_chunks("abcdefghijklm",3)
    ['abcde', 'fghij', 'klm']
    """
    return list(in_chunks_iter(inputs,num_chunks))

def in_chunks_iter(inputs,num_chunks:int):
    chunk_size = math.ceil(len(inputs) / num_chunks)
    for chunk in range(num_chunks):
        chunk = inputs[chunk * chunk_size : (chunk+1)*chunk_size]
        yield chunk

def run_in_parallel(n_jobs:int,inputs,callable,):
    """
    Executes the given task in parallel.

    >>> run_in_parallel(3,list(range(1000)),lambda lst: list(map(lambda i:i**3,lst)))[0:10]
    [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
    >>> run_in_parallel(4,list(range(200)),lambda lst: list(map(lambda i:i**3,lst)))[100]
    1000000
    """
    chunks = in_chunks(inputs,n_jobs)
    return flatten_once(Parallel(n_jobs=n_jobs)(delayed(callable)(chunk) for chunk in chunks))

def tile_images(imgs:List[Image],tile_w:int,tile_h:int,n_tiles_w:int,n_tiles_h:int,with_replacement=False,) -> Image:
    """
    Tiles the given images into one to make for a convenient quick display of datapoints.
    :param imgs:
    :param tile_w:
    :param tile_h:
    :param n_tiles_w:
    :param n_tiles_h:
    :param with_replacement:
    :return:
    """
    imgs_left = list(range(len(imgs)))
    w_total = n_tiles_w * tile_w
    h_total = n_tiles_h * tile_h
    tile_image = PIL.Image.new('RGB', (w_total, h_total))
    x = 0
    y = 0
    while y < h_total:
        while x < w_total:
            idx = random.choice(imgs_left)
            if not with_replacement:
                imgs_left = [i for i in imgs_left if i != idx]
            img = imgs[idx]
            tile_image.paste(img.resize((tile_w,tile_h)),(x,y))

            x += tile_w
        x = 0
        y += tile_h

    return tile_image

def tile_images_with_annots__(imgs:List[Image],
    annots:List[str],
    tile_w:int,tile_h:int,
    n_tiles_w:int,n_tiles_h:int,
    with_replacement=False,
    annot_color=(0,0,0),
                            ) -> Image:
    imgs_left = list(range(len(imgs)))
    w_total = n_tiles_w * tile_w
    h_total = n_tiles_h * tile_h
    tile_image = PIL.Image.new('RGB', (w_total, h_total))
    draw = ImageDraw.Draw(tile_image)
    x = 0
    y = 0
    while y < h_total:
        while x < w_total:
            idx = random.choice(imgs_left)
            if not with_replacement:
                imgs_left = [i for i in imgs_left if i != idx]
            img = imgs[idx]
            tile_image.paste(img.resize((tile_w,tile_h)),(x,y))
            annot = annots[idx]
            draw.text((tile_w,tile_h),annot,annot_color)
            x += tile_w
        x = 0
        y += tile_h

    return tile_image




def tile_images_with_annots(imgs: List,
                            annots: List[str],
                            tile_w: int, tile_h: int,
                            n_tiles_w: int, n_tiles_h: int,
                            with_replacement=False,
                            annot_color=(0, 0, 0),
                            ) -> Image:
    imgs_left = list(range(len(imgs)))
    w_total = n_tiles_w * tile_w
    h_total = n_tiles_h * tile_h
    tile_image = PIL.Image.new('RGB', (w_total, h_total))
    draw = ImageDraw.Draw(tile_image)
    x = 0
    y = 0
    while y < h_total:
        while x < w_total:
            if imgs_left:
                idx = random.choice(imgs_left)

                img = imgs[idx]

            else:
                img = PIL.Image.new('RGB', (16, 16), color='white')  # will be resized anyways
            tile_image.paste(img.resize((tile_w, tile_h)), (x, y))

            if imgs_left:
                annot = annots[idx]
                draw.text((x, y), annot, annot_color)

                if not with_replacement:
                    imgs_left = [i for i in imgs_left if i != idx]

            x += tile_w
        x = 0
        y += tile_h

    return tile_image


def merge_dataframes_per_cell(dfs,):
    """
    >>> df1 = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
    >>> df2 = pd.DataFrame({"a":[11,22,33],"b":[4,5,6]})
    >>> merge_dataframes_per_cell([df1,df2,]) # doctest:+NORMALIZE_WHITESPACE
        a       b
    0  [1, 11]  [4, 4]
    1  [2, 22]  [5, 5]
    2  [3, 33]  [6, 6]
    """
    df_shapes = {df.values.shape for df in dfs}
    if len(df_shapes) > 1:
        raise ValueError(f"Expected dataframes of same shape but found: {df_shapes}")
    
    rslt = []
    df1 = dfs[0]
    for idx in df1.index:
        row_dct = defaultdict(lambda: [])
        for df in dfs:
            row = df.loc[idx]
            for col in df.columns:
                row_dct[col].append(row[col])
        rslt.append(row_dct)
    return pd.DataFrame(rslt)


def mu_sigma_reduction(df):
    """
    For cells that contain several distinct elements,
    averages and standard deviations are computed.
    For lists containing a single value, that value
    is extracted.
    This is useful for computing stats on dataframes
    that contain samples as lists as they are 
    produced by the function merge_dataframes_per_cell
    defined above.

    >>> df = pd.DataFrame({"a":[[10,15],[20,30]],"b":["q","r"]})
    >>> mu_sigma_reduction(df) # doctest:+NORMALIZE_WHITESPACE
        a__mu  a__std  b
    0   12.5     2.5  q
    1   25.0     5.0  r
    """
    
    rslt = []
    for idx in df.index:
        row_dct = {}
        row = df.loc[idx]
        for col in df.columns:
            if isinstance(row[col],list):
                if len(set(row[col])) == 1:
                    row_dct[col] = row[col][0]
                    if isinstance(row[col][0],float):
                        row_dct[col+"__mu"] = np.array(row[col]).mean()
                        row_dct[col+"__std"] = np.array(row[col]).std()
                elif len(set(row[col])) == 0:
                    row_dct[col] = []
                else:
                    row_dct[col+"__mu"] = np.array(row[col]).mean()
                    row_dct[col+"__std"] = np.array(row[col]).std()
            else:
                row_dct[col] = row[col]

        rslt.append(row_dct)
    return pd.DataFrame(rslt)


def german_to_english_nums(s:str)->str:
    """
    Given a string s containing german numbers,
    this method will replace any occurrence of
    german numbers by the english equivalent.

    Be careful as english numbers use the comma
    with another intended meaning of separating 
    thousand places...

    >>> german_to_english_nums("1,23 ")
    '1.23 '
    >>> german_to_english_nums(
    ... 'Take 1,23 g of something then add 1.23 mL to it.'
    ... )
    'Take 1.23 g of something then add 1.23 mL to it.'
    >>> german_to_english_nums("1,1-dichloromethane")
    '1,1-dichloromethane'
    >>> german_to_english_nums("0,4:1")
    '0.4:1'
    >>> german_to_english_nums("0,9:1,5")
    '0.9:1.5'
    """
    s = re.sub(r'([0-9]+)\,([0-9]+\s)', r'\1.\2', s)
    s = re.sub(r'(\s[0-9]+)\,([0-9]+)', r'\1.\2', s)
    s = re.sub(r'([0-9]+)\,([0-9]+:)', r'\1.\2', s)
    s = re.sub(r'(:[0-9]+)\,([0-9]+)', r'\1.\2', s)
    return s