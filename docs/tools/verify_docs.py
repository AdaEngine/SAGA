#!/usr/bin/env python3
"""Current docs/policy verification; historical report is never rewritten."""
import json, os, pathlib, unittest
from runtime_policy import audit, unexpected_project_files
ROOT=pathlib.Path(__file__).resolve().parents[2]
def main():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'docs/tools'),pattern='test*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0: raise SystemExit(1)
    report=audit()
    report['documentation_tests']={'discovered':result.testsRun,'executed':result.testsRun-len(result.skipped),'skipped':[{'test':str(test),'reason':reason} for test,reason in result.skipped],'failures':len(result.failures),'errors':len(result.errors),'reference_enabled':bool(os.environ.get('SAGA_REFERENCE'))}
    report['executed']={'swift_build':False,'swift_tests':False,'gpu':False,'cts':False,'shader_translation':False}
    output=ROOT/'.verification/docs-current.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
