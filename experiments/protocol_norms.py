"""Independent exact rational interpretation of the pinned Pearl norm contract."""
from fractions import Fraction
import struct


def bf16_value(bits):
    sign=-1 if bits & 0x8000 else 1
    bits &= 0x7fff
    exponent,fraction=bits>>7,bits&127
    if exponent==255:raise ValueError('Non-finite BF16')
    return sign*Fraction(fraction+(128 if exponent else 0))*Fraction(2)**(max(exponent,1)-134)


def round_bf16(value):
    if value<0:return round_bf16(-value)|0x8000
    low,high=0,0x7f7f
    while low<high:
        mid=(low+high+1)//2
        if bf16_value(mid)<=value:low=mid
        else:high=mid-1
    if bf16_value(low)==value:return low
    if low==0x7f7f:raise ValueError('BF16 overflow')
    midpoint=(bf16_value(low)+bf16_value(low+1))/2
    return low+int(value>midpoint or (value==midpoint and low&1))


def sqrt_bf16(value):
    assert value>=0
    low,high=0,0x7f7f
    while low<high:
        mid=(low+high+1)//2
        if bf16_value(mid)**2<=value:low=mid
        else:high=mid-1
    if bf16_value(low)**2==value:return low
    assert low<0x7f7f
    midpoint=(bf16_value(low)+bf16_value(low+1))/2
    return low+int(value>midpoint**2 or (value==midpoint**2 and low&1))


def exact_norms(values,scales):
    k=len(values)
    assert k and k%8==0 and len(scales)==k//8 and k<=65536
    width=27-(k-1).bit_length()
    terms=[];largest=Fraction(0)
    for start,code in zip(range(0,k,8),scales):
        scale=abs(bf16_value(int(code)))
        block=[int(x) for x in values[start:start+8]]
        largest=max(largest,max(abs(x) for x in block)*scale)
        unsigned=int(code)&0x7fff
        exponent=unsigned>>7
        mantissa=(unsigned&127)+(128 if exponent else 0)
        product=mantissa**2*sum(x*x for x in block)
        if product:terms.append((product,2*max(exponent,1)))
    if terms:
        maximum=max(e for _,e in terms)
        frame=sum((p<<width)>>(maximum-e) for p,e in terms)
        mean=Fraction(frame,k)*Fraction(2)**(maximum-268-width)
        code=(sqrt_bf16(mean)+2)&0xfffc
    else:code=0
    assert code<0x7f80
    return code,round_bf16(largest)


def scales_from_norms(l2_bits,linf_bits):
    floor=Fraction(1,1<<32)
    l2=max(bf16_value(l2_bits),floor);linf=max(bf16_value(linf_bits),floor)
    # BF16 rounding of sqrt(32)/2 and its 2^-16 multiple, independently rational-bracketed.
    delta_r=bf16_value(sqrt_bf16(Fraction(8)))
    delta_over=bf16_value(sqrt_bf16(Fraction(8,1<<32)))
    bound=bf16_value(round_bf16(delta_r*l2+linf))
    alpha=round_bf16(Fraction(448)/bound)
    alpha_l2=bf16_value(round_bf16(bf16_value(alpha)*l2))
    beta=round_bf16(alpha_l2*delta_over)
    return alpha,beta


def fixture(path):
    raw=path.read_bytes()
    assert len(raw)==3328
    values=list(struct.unpack('<1024b',raw[:1024]))
    scales=list(struct.unpack('<128H',raw[1024:1280]))
    original=list(struct.unpack('<1024H',raw[1280:]))
    return values,scales,original


if __name__=='__main__':
    from pathlib import Path
    assert __debug__
    values,scales,_=fixture(Path('fixtures/norm-boundary.bin'))
    norms=exact_norms(values,scales)
    assert norms[0]==0x3f80
    assert scales_from_norms(*norms)==(0x4291,0x3b4d)
    assert exact_norms([0]*1024,[0]*128)==(0,0)
    for code in (1,127,128,0x3f80,0x3f81,0x7f7f):
        assert round_bf16(bf16_value(code))==code
        assert sqrt_bf16(bf16_value(code)**2)==code
    print('Exact rational norm and scale-chain checks passed')
