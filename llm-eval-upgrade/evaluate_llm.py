#!/usr/bin/env python3
"""Black-box LLM QA evaluation against the running RAG FastAPI; stdlib only.
Reports exact fact checks and provenance, not human-judged correctness/groundedness.
"""
import argparse, json, statistics, time, urllib.request, urllib.error
from pathlib import Path

# Fictional, independent documents ensure no real company data is used.
CASES = [
 ('orion_remote','At Orion Labs, staff can work from home four days per week.', 'How many days each week are Orion Labs employees allowed to work remotely?', ['four','days'], 'answered'),
 ('orion_leave','At Orion Labs, the annual vacation entitlement is 27 paid days.', 'How much paid annual holiday do Orion Labs staff get?', ['27','days'], 'answered'),
 ('orion_vpn','Orion Labs staff must connect using the SkyShield VPN when accessing internal applications offsite.', 'Which VPN is mandatory for remote access to internal Orion Labs applications?', ['skyshield'], 'answered'),
 ('orion_expenses','At Orion Labs, expenses over 175 euros require approval from a department manager.', 'Who needs to approve an Orion Labs reimbursement above 175 euros?', ['manager'], 'answered'),
 ('orion_training','New Orion Labs employees must finish data privacy training within 21 days of their start date.', 'What is the Orion Labs deadline for onboarding privacy training?', ['21','days'], 'answered'),
 ('orion_retention','Orion Labs keeps closed support tickets for 15 months.', 'How long are resolved customer support tickets stored at Orion Labs?', ['15','months'], 'answered'),
 ('orion_backup','Orion Labs production databases are backed up every six hours.', 'How frequently does Orion Labs save production database backups?', ['six','hours'], 'answered'),
 ('orion_security','Orion Labs security incidents must be reported within 90 minutes.', 'What is the deadline for reporting an Orion Labs security incident?', ['90','minutes'], 'answered'),
 ('orion_delivery','Orion Labs standard orders are delivered within eight business days.', 'What delivery time does Orion Labs promise for standard orders?', ['eight','business days'], 'answered'),
 ('orion_warranty','Orion Labs devices have a three-year limited warranty.', 'How long does the Orion Labs device warranty last?', ['three','year'], 'answered'),
 ('orion_password','Orion Labs passwords must be at least 16 characters long.', 'What is the minimum Orion Labs password length?', ['16','character'], 'answered'),
 ('orion_approval','At Orion Labs, a director must authorize international business trips before booking.', 'Who signs off on overseas Orion Labs business travel?', ['director'], 'answered'),
 ('orion_parking','Orion Labs visitor parking passes must be requested 48 hours in advance.', 'How early should visitors request parking permits at Orion Labs?', ['48','hour'], 'answered'),
 ('orion_meals','The maximum Orion Labs meal allowance is 14 euros per workday.', 'What is the Orion Labs daily lunch subsidy cap?', ['14','euro'], 'answered'),
 ('orion_reviews','Formal Orion Labs performance reviews happen twice a year.', 'How frequently are employee appraisals carried out at Orion Labs?', ['twice','year'], 'answered'),
 ('orion_notice','Orion Labs announces planned maintenance at least 96 hours beforehand.', 'How much advance notice is required before Orion Labs scheduled maintenance?', ['96','hour'], 'answered'),
 ('orion_budget','At Orion Labs, purchases above 1250 euros must be approved by the finance team.', 'Who approves Orion Labs purchases greater than 1250 euros?', ['finance'], 'answered'),
 ('orion_archive','Orion Labs archives financial statements for nine years.', 'How long does Orion Labs preserve historical finance reports?', ['nine','year'], 'answered'),
]


def request_json(method, url, token, data=None, timeout=90):
    encoded = None if data is None else json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=encoded, method=method, headers={
        'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def evaluate_case(case, result):
    doc_id, _, _, needles, desired_status = case
    answer = str(result.get('answer','')).lower()
    status = result.get('status')
    citations = result.get('citations') or []
    cited_docs = [x.get('document_id') for x in citations if isinstance(x,dict)]
    # Alternative numeral is acceptable where shown; checks remain heuristic.
    checks = []
    for needle in needles:
        variants = {'four':['four','4'],'six':['six','6'],'eight':['eight','8'],
                    'three':['three','3'],'twice':['twice','two times','2 times'],
                    'nine':['nine','9']}.get(needle,[needle])
        checks.append(any(term in answer for term in variants))
    return {'status_correct':status == desired_status,
            'expected_fact_present':all(checks),
            'expected_source_cited':doc_id in cited_docs,
            'generation_mode': result.get('generation_mode','missing'),
            'latency_ms': result.get('latency_ms'),
            'answer': result.get('answer'),
            'cited_documents':cited_docs}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--token',default='demo-orbit',help='Use demo-orbit to avoid altering demo-acme docs')
    p.add_argument('--output',default='reports/llm_evaluation.json')
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--timeout',type=int,default=90)
    args=p.parse_args()
    base=args.base_url.rstrip('/')
    try:
        health=request_json('GET',f'{base}/health',args.token,timeout=5)
    except Exception as err:
        raise SystemExit(f'API unreachable at {base}/health: {err}')
    print('API health:',health,flush=True)
    print('Ingesting 18 synthetic documents...',flush=True)
    for doc, content, *_ in CASES:
        try:
            request_json('POST',f'{base}/documents',args.token,{'document_id':doc,'content':content},args.timeout)
        except (urllib.error.URLError,ValueError) as err:
            raise SystemExit(f'Ingestion failed for {doc}: {err}')
    records=[]
    for index,case in enumerate(CASES,1):
        doc,_,question,_,_ = case
        started=time.perf_counter()
        try:
            response=request_json('POST',f'{base}/ask',args.token,{'question':question,'top_k':args.top_k},args.timeout)
            score=evaluate_case(case,response)
            score.update({'id':doc,'question':question,'elapsed_wall_ms':round((time.perf_counter()-started)*1000,2)})
        except Exception as err:
            score={'id':doc,'question':question,'error':str(err),'generation_mode':'error',
                   'status_correct':False,'expected_fact_present':False,'expected_source_cited':False}
        records.append(score)
        print(f'{index:02d}/{len(CASES)} {doc}: {score["generation_mode"]} fact={score["expected_fact_present"]} cited={score["expected_source_cited"]}',flush=True)
    def rate(key): return round(sum(bool(r.get(key)) for r in records)/len(records),4)
    times=[r['latency_ms'] for r in records if isinstance(r.get('latency_ms'),(float,int))]
    modes={mode:sum(r.get('generation_mode')==mode for r in records) for mode in ['llm','fallback','not_attempted','missing','error']}
    summary={'queries':len(records),'generation_modes':modes,'llm_success_rate':round(modes['llm']/len(records),4),
             'fact_heuristic_pass_rate':rate('expected_fact_present'),
             'expected_citation_rate':rate('expected_source_cited'),
             'status_accuracy':rate('status_correct'),
             'mean_api_latency_ms':round(statistics.mean(times),2) if times else None,
             'median_api_latency_ms':round(statistics.median(times),2) if times else None,
             'caveat':'18 synthetic positive questions; automatic phrase-based checks only. This is not a full groundedness/hallucination audit. API generation_mode reflects project instrumentation.'}
    report={'summary':summary,'cases':records}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('\n'+json.dumps(summary,indent=2)+'\nSaved '+str(output))

if __name__=='__main__': main()
