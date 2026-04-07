import os
import torch
import numpy as np
from torch.utils.cpp_extension import load


PRECISION = 16  # DO NOT EDIT!


# Load on-the-fly with ninja.
torchac_dir = os.path.dirname(os.path.realpath(__file__))
backend_dir = os.path.join(torchac_dir, 'backend')
numpyAc_backend = load(
  name="numpyAc_backend",
  sources=[os.path.join(backend_dir, "numpyAc_backend.cpp")],
  verbose=False)

def encode_float_cdf(cdf_float,
                     sym,
                     needs_normalization=True,
                     check_input_bounds=False):
  """Encode symbols `sym` with potentially unnormalized floating point CDF.

  Check the README for more details.

  :param cdf_float: CDF tensor, float32, on CPU. Shape (N1, ..., Nm, Lp).
  :param sym: The symbols to encode, int16, on CPU. Shape (N1, ..., Nm).
  :param needs_normalization: if True, assume `cdf_float` is un-normalized and
    needs normalization. Otherwise only convert it, without normalizing.
  :param check_input_bounds: if True, ensure inputs have valid values.
    Important: may take significant time. Only enable to check.

  :return: byte-string, encoding `sym`.
  """
  if check_input_bounds:
    if cdf_float.min() < 0:
      raise ValueError(f'cdf_float.min() == {cdf_float.min()}, should be >=0.!')
    if cdf_float.max() > 1:
      raise ValueError(f'cdf_float.max() == {cdf_float.max()}, should be <=1.!')
    Lp = cdf_float.shape[-1]
    if sym.max() >= Lp - 1:
      raise ValueError(f'sym.max() == {sym.max()}, should be <=Lp - 1.!')
  cdf_int = quantize_values(cdf_float, needs_normalization)
  return encode_int_cdf(cdf_int, sym)


def encode_int_cdf(cdf_int, sym):
  """Encode symbols `sym` with a normalized integer cdf `cdf_int`.

  Check the README for more details.

  :param cdf_int: CDF tensor, int16, on CPU. Shape (N1, ..., Nm, Lp).
  :param sym: The symbols to encode, int16, on CPU. Shape (N1, ..., Nm).

  :return: byte-string, encoding `sym`
  """
  cdf_int, sym = validate_input(cdf_int, sym)
  return numpyAc_backend.encode_cdf( torch.ShortTensor(cdf_int), torch.ShortTensor(sym))


def validate_input(cdf, sym=None):
  """Check device, dtype, and shapes."""
  if sym is not None and sym.dtype != np.int16:
    raise ValueError('Symbols must be int16!')
  if sym is not None:
    if len(cdf.shape) != len(sym.shape) + 1 or cdf.shape[:-1] != sym.shape:
      raise ValueError(f'Invalid shapes of cdf={cdf.shape}, sym={sym.shape}! '
                       'The first m elements of cdf.shape must be equal to '
                       'sym.shape, and cdf should only have one more dimension.')
  Lp = cdf.shape[-1]
  cdf = cdf.reshape(-1, Lp)
  if sym is None:
    return cdf
  sym = sym.reshape(-1)
  return cdf, sym


def quantize_values(cdf_float, needs_normalization):
  r"""Convert floatingpoint CDF to integers. See README for more info.

  The idea is the following:
  When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
  (note that 1 should not be included.)
  We now want to convert this to int16 but make sure we do not get
  the same value twice, as this would break the arithmetic coder
  (you need a strictly monotonically increasing function).
  So, if needs_normalization==True, we multiply the input CDF
  with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
  Then, in a final step, we add an arange(Lp), which is just a line with
  slope one. This ensure that for sure, we will get unique, strictly
  monotonically increasing CDFs, which are \in [0, 2**16)
  """
  Lp = cdf_float.shape[-1]
  factor = 2**PRECISION
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float*(new_max_value)
  cdf_float = np.round(cdf_float)
  cdf = cdf_float.astype(np.int16)
  if needs_normalization:
    r = np.arange(Lp)
    cdf+=r
  return cdf

def pdf_to_cdf_norm(pdf):
    assert pdf.ndim==2
    cdfF = np.cumsum( pdf, axis=1)
    cdfF = cdfF/cdfF[:,-1:]
    cdfF = np.hstack((np.zeros((pdf.shape[0],1)),cdfF))
    return cdfF

class arithmeticCoding():
  def __init__(self) -> None:
    self.byte_stream = None


  def compress_bitstream(self, pdf, sym, binfile=None):
    """
    Encode symbols using arithmetic coding.

    Args:
        pdf: Probability distribution (N, symDim)
        sym: Symbols to encode (N,)
        binfile: Optional path to write compressed bitstream

    Returns:
        Tuple of (byte_stream, real_bits)
    """
    assert pdf.shape[0] == sym.shape[0]
    assert pdf.ndim == 2 and sym.ndim == 1

    cdfF = pdf_to_cdf_norm(pdf)
    self.byte_stream = encode_float_cdf(cdfF, sym, check_input_bounds=True)
    real_bits = len(self.byte_stream) * 8

    if binfile is not None:
      with open(binfile, 'wb') as fout:
          fout.write(self.byte_stream)
    return self.byte_stream, real_bits

class arithmeticDeCoding():
  """
  Arithmetic decoding class.

  Args:
      byte_stream: Binary file stream
      sysNum: Number of symbols to decode (must be stored externally)
      symDim: Number of possible symbols
      binfile: Optional path to bin file for loading byte_stream
  """

  def __init__(self, byte_stream, sysNum, symDim, binfile=None) -> None:
    if binfile is not None:
      with open(binfile, 'rb') as fin:
        byte_stream = fin.read()
    self.byte_stream = byte_stream
    self.decoder = numpyAc_backend.decompress_bitstream(self.byte_stream, sysNum, symDim+1)

  def decompress_bitstream(self, pdf):
    """
    Decode symbols using arithmetic decoding.

    Args:
        pdf: Probability distribution (N, symDim)

    Returns:
        Decoded symbols
    """
    cdfF = pdf_to_cdf_norm(pdf)
    pro = quantize_values(cdfF, needs_normalization=True)
    pro = pro.squeeze(0).astype(np.uint16).tolist()
    sym_out = self.decoder.decodeAsym(pro)
    return sym_out
