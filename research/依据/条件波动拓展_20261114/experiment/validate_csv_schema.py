"""Validate frozen results with the CSV block column explicitly read as text.

The original fitted model code and protocol remain unchanged. Pandas can infer
the numeric-looking period label '2026' as int in some chunks, so this validation
entry point supplies the schema before calling the frozen numerical validator.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import pandas as pandas_module

ROOT=Path(__file__).resolve().parents[3]
FIT_SOURCE=ROOT/'工作文件/条件波动拓展_20261114/extension_fit.py'
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class ExplicitSchemaPandas:
    def __getattr__(self,name):return getattr(pandas_module,name)
    def read_csv(self,path,*args,**kwargs):
        if str(path).endswith('.point_scores.csv.gz'):
            kwargs['dtype']={**kwargs.get('dtype',{}),'trading_date':str,'block':str}
            kwargs['low_memory']=False
        return pandas_module.read_csv(path,*args,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--symbols');args=p.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_extension',FIT_SOURCE)
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit);fit.init()
    protected=[FIT_SOURCE,OUT/'protocol.json',OUT/'selection.json',*sorted((OUT/'training').glob('*.json')),
               *sorted(OUT.glob('*.evaluation.json')),*sorted(OUT.glob('*.point_scores.csv.gz'))]
    before={str(x):sha(x) for x in protected}
    fit.pd=ExplicitSchemaPandas()
    symbols=args.symbols.split(',') if args.symbols else fit.symbols()
    for symbol in symbols:
        result=fit.validate(symbol)
        result['reader_schema']={'trading_date':'str','block':'str','low_memory':False}
        result['validation_entry_sha256']=sha(Path(__file__))
        fit.dump(OUT/f'{symbol}.validation.json',result)
    assert before=={str(x):sha(x) for x in protected}
    print('CSV SCHEMA VALIDATION PASSED',symbols,'protected inputs unchanged',flush=True)

if __name__=='__main__':main()
