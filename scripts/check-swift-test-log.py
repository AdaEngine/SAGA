#!/usr/bin/env python3
"""Do not mistake XCTest's empty compatibility runner for actual Swift Testing."""
import pathlib,re,sys

def require_nonzero_success(text):
    counts=re.findall(r'Test run with (\d+) tests?\b[^\n]*passed',text)
    if not counts or not any(int(n)>0 for n in counts): raise ValueError('Missing nonzero successful Swift Testing summary')
    if re.search(r'Test run .*failed|recorded an issue',text): raise ValueError('Swift Testing reported a failure')
    return max(map(int,counts))
if __name__=='__main__': print(f'Swift Testing verified: {require_nonzero_success(pathlib.Path(sys.argv[1]).read_text())} tests')
