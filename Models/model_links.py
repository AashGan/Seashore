# Model imports and dictonary for convenience
from .casum.model import CA_SUM
from .csta.model_routing import set_model
from .ctv.ctv_utils import ctv_functional
from .mars.model import MARs
from .pgl_sum.summarizer import PGL_SUM
from .vasnet.vasnet import VASNet
from .sumgda.model import SUM_GDA
from .mrhisum.sl_module import SL_module
from .videoxum.model import VTSum_BLIP_TT,VTSum_BLIP_TT_CA
from .sd_vsum import SD_VSUM
model_dict = {'casum':CA_SUM,'ctv_raw':ctv_functional,'mars':MARs,'pgl_sum':PGL_SUM,'vasnet':VASNet,
              'sl-module':SL_module,'sumgda':SUM_GDA,'csta':set_model,'vtsum_blip_tt':VTSum_BLIP_TT,
              'vtsum_blip_tt_ca':VTSum_BLIP_TT_CA,'sd_vsum':SD_VSUM}