"""Build the actual pinned upstream norm functions as a small CPU executable.

Upstream functions remain unchanged. Unrelated proof-system code is not linked.
Generated source and upstream license are kept under the ignored build directory.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

REVISION='b7160514ed3d862fb48e850ba9a70a545cefcb1e'


def item(source,pattern):
    match=re.search(pattern,source)
    assert match,pattern
    opening=source.index('{',match.start())
    depth=1;position=opening+1
    while depth:
        if source[position]=='{':depth+=1
        if source[position]=='}':depth-=1
        position+=1
    return source[match.start():position]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=pathlib.Path)
    args=parser.parse_args()
    upstream=args.upstream
    assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    files={name:subprocess.check_output(['git','-C',str(upstream),'show',REVISION+':'+path]).decode() for name,path in {
        'prequant':'zk-pow/src/v4/api/prequant.rs','dtype':'zk-pow/src/v4/api/dtype.rs',
        'sqrt':'zk-pow/src/v4/circuit/scale_stark/stark.rs'}.items()}
    pieces=['use anyhow::{Result,Context,ensure};','use core::cmp::Ordering;',
            'const L2_ROUNDED_BITS:u32=2;']
    for source,names in [('prequant',['round_l2_to_grid','l2_frame_width','exact_norms','row_norms']),
                         ('dtype',['f32_to_bf16','bf16_to_f32','check_not_nan_or_inf_bf16'])]:
        for name in names:
            pieces.append(item(files[source],r'(?:pub(?:\([^)]*\))?\s+)?fn '+name+r'\b'))
    pieces.append(item(files['sqrt'],r'pub\(crate\) struct SqrtClaim\b'))
    pieces.append(item(files['sqrt'],r'impl SqrtClaim\b'))
    for name in ['cmp_shifted','sqrt_bracket_ok','rne_sqrt_hat']:
        pieces.append(item(files['sqrt'],r'(?:pub(?:\([^)]*\))?\s+)?fn '+name+r'\b'))
    pieces.append(r'''
fn main(){
 let path=std::env::args().nth(1).expect("input file");let raw=std::fs::read(path).unwrap();
 let m=u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
 let k=u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
 assert_eq!(raw.len(),8+m*k+m*k/8*2);
 let ints:Vec<i8>=raw[8..8+m*k].iter().map(|x|*x as i8).collect();
 let scales:Vec<u16>=raw[8+m*k..].chunks_exact(2).map(|x|u16::from_le_bytes([x[0],x[1]])).collect();
 let norms=exact_norms(&ints,&scales,m,k,8).unwrap();
 println!("[{}]",norms.iter().map(|(a,b)|format!("[{},{}]",a,b)).collect::<Vec<_>>().join(","));
}
''')
    build=pathlib.Path('build/rust-norms');(build/'src').mkdir(parents=True,exist_ok=True)
    (build/'Cargo.toml').write_text('[package]\nname="requant-upstream-norms"\nversion="0.1.0"\nedition="2024"\n[dependencies]\nanyhow="=1.0.100"\n',encoding='utf-8')
    (build/'src/main.rs').write_text('\n\n'.join(pieces)+'\n',encoding='utf-8')
    (build/'UPSTREAM-LICENSE').write_bytes((upstream/'LICENSE').read_bytes())
    subprocess.run(['cargo','build','--release','--manifest-path',str(build/'Cargo.toml')],check=True)
    metadata=dict(revision=REVISION,rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
                  source_sha256={'build_rust_norms.py':hashlib.sha256(pathlib.Path('build_rust_norms.py').read_bytes()).hexdigest()},
                  upstream_source_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()},
                  generated_source_sha256=hashlib.sha256((build/'src/main.rs').read_bytes()).hexdigest(),
                  scope='Unchanged upstream exact_norms, dtype conversion and sqrt/bracket functions; isolated from the rest of the verifier. This does not execute a full plain-proof or ZK-certificate verification.')
    pathlib.Path('results/rust-norm-build.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
