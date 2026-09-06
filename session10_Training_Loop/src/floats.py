"""Writing a decimal number out in a floating point format, by hand.

``encode`` does the arithmetic the way you would on paper -- find the power of
two, read off the significand, round the mantissa to the bits available with
round-to-nearest-ties-to-even -- using exact rational arithmetic throughout, so
nothing here inherits an answer from the hardware it is meant to be explaining.
Every result is then cross-checked against the real bit pattern from torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Format:
    name: str
    exp_bits: int
    mant_bits: int
    note: str = ""

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def total_bits(self) -> int:
        return 1 + self.exp_bits + self.mant_bits

    @property
    def max_exp_field(self) -> int:
        # E4M3 (the OCP "fn" variant used for training) has no infinities: the
        # all-ones exponent is a normal exponent and only S.1111.111 is NaN, so
        # its usable exponent field runs one higher than IEEE's would.
        return (1 << self.exp_bits) - 1

    @property
    def min_normal_exp(self) -> int:
        return 1 - self.bias


FP32 = Format("fp32", 8, 23, "IEEE 754 binary32")
FP16 = Format("fp16", 5, 10, "IEEE 754 binary16")
BF16 = Format("bf16", 8, 7, "fp32's exponent, fp32's top 16 bits")
FP8_E4M3 = Format("fp8 E4M3", 4, 3, "OCP e4m3fn — no infinities, max 448")
FP8_E5M2 = Format("fp8 E5M2", 5, 2, "OCP e5m2 — IEEE-like, has inf/NaN")
FP4_E2M1 = Format("fp4 E2M1", 2, 1, "NVFP4 element, always block-scaled")


@dataclass
class Encoded:
    fmt: Format
    sign: int
    exp_field: int
    mantissa: int
    unbiased_exp: int
    subnormal: bool
    value: Fraction
    target: Fraction

    @property
    def bits(self) -> str:
        return (f"{self.sign:01b} "
                f"{self.exp_field:0{self.fmt.exp_bits}b} "
                f"{self.mantissa:0{self.fmt.mant_bits}b}")

    @property
    def bits_flat(self) -> str:
        return self.bits.replace(" ", "")

    @property
    def hex(self) -> str:
        n = int(self.bits_flat, 2)
        return f"0x{n:0{(self.fmt.total_bits + 3)//4}X}"

    @property
    def significand_binary(self) -> str:
        lead = "0" if self.subnormal else "1"
        return f"{lead}.{self.mantissa:0{self.fmt.mant_bits}b}"

    @property
    def abs_error(self) -> Fraction:
        return abs(self.value - self.target)

    @property
    def rel_error(self) -> Fraction:
        return self.abs_error / abs(self.target) if self.target else Fraction(0)

    @property
    def ulp(self) -> Fraction:
        """The spacing between representable numbers at this magnitude."""
        e = self.fmt.min_normal_exp if self.subnormal else self.unbiased_exp
        return Fraction(2) ** (e - self.fmt.mant_bits)

    def decimal(self, places: int = 30) -> str:
        return f"{float(self.value):.{places}f}".rstrip("0")


def _floor_log2(x: Fraction) -> int:
    e = 0
    while x >= 2:
        x /= 2
        e += 1
    while x < 1:
        x *= 2
        e -= 1
    return e


def encode(x, fmt: Format) -> Encoded:
    """Round ``x`` into ``fmt``, ties to even, in exact rational arithmetic."""
    target = Fraction(x)
    sign = 1 if target < 0 else 0
    a = abs(target)
    if a == 0:
        return Encoded(fmt, sign, 0, 0, fmt.min_normal_exp, True,
                       Fraction(0), target)

    e = _floor_log2(a)
    subnormal = e < fmt.min_normal_exp
    if subnormal:
        e = fmt.min_normal_exp

    # significand in [1, 2) for normals, in [0, 1) for subnormals
    sig = a / (Fraction(2) ** e)
    scaled = sig * (1 << fmt.mant_bits)          # exact rational
    lo = scaled.numerator // scaled.denominator  # floor
    frac = scaled - lo
    if frac > Fraction(1, 2):
        rounded = lo + 1
    elif frac < Fraction(1, 2):
        rounded = lo
    else:                                        # exact tie -> to even
        rounded = lo + (lo & 1)

    if subnormal:
        mant = rounded
        if mant >= (1 << fmt.mant_bits):         # rounded up into normal range
            subnormal = False
            mant -= (1 << fmt.mant_bits)
            exp_field = 1
            e = fmt.min_normal_exp
        else:
            exp_field = 0
    else:
        if rounded >= (2 << fmt.mant_bits):      # carried out of [1, 2)
            rounded >>= 1
            e += 1
        mant = rounded - (1 << fmt.mant_bits)
        exp_field = e + fmt.bias
        if exp_field > fmt.max_exp_field:
            raise OverflowError(f"{x} overflows {fmt.name}")

    lead = 0 if subnormal else 1
    value = (Fraction(lead) + Fraction(mant, 1 << fmt.mant_bits)) * \
        (Fraction(2) ** e)
    if sign:
        value = -value
    return Encoded(fmt, sign, exp_field, mant, e, subnormal, value, target)


def binary_expansion(x: float, digits: int = 32) -> str:
    """The long-division a person would do: repeatedly double and read off."""
    a = Fraction(x)
    ip = int(a)
    a -= ip
    out = [f"{ip:b}", "."]
    for _ in range(digits):
        a *= 2
        bit = int(a)
        out.append(str(bit))
        a -= bit
    return "".join(out) + ("…" if a else "")


def torch_bits(x: float, dtype) -> str | None:
    """The real bit pattern the hardware stores, for cross-checking."""
    import torch

    try:
        t = torch.tensor([x], dtype=dtype)
    except (RuntimeError, TypeError):
        return None
    nbytes = t.element_size()
    raw = t.view(torch.uint8).numpy().tobytes()          # little-endian
    n = int.from_bytes(raw, "little")
    return f"{n:0{nbytes*8}b}"
