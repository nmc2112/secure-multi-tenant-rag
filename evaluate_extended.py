"""Standalone retrieval benchmark: relevance, abstention, MRR, latency, isolation.
Run from the project root; dependencies are those of the existing project.
Semantic Qdrant storage must be separate from the running API instance.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from app.core import LexicalStore, SemanticStore

# A deliberately small, readable synthetic evaluation corpus. No claims of generalization.
DATA = [
    ('remote', 'Employees may work remotely up to three days in each working week.', 'How many days may staff work away from the office?'),
    ('leave', 'Full-time staff receive 25 days of paid annual leave per year.', 'What is the yearly vacation allowance?'),
    ('expenses', 'Expense reimbursements above 150 euros require manager approval.', 'Who must sign off on larger reimbursements?'),
    ('security', 'All employees must enable multi-factor authentication on company accounts.', 'Is a second sign-in factor mandatory?'),
    ('vpn', 'Access to internal systems outside the office requires the company VPN.', 'How do I securely connect to internal services from home?'),
    ('laptop', 'Company laptops must be returned within seven calendar days after employment ends.', 'When do departing employees give back their computers?'),
    ('training', 'Every new hire must complete privacy training within 30 days of joining.', 'What is the onboarding deadline for data protection education?'),
    ('retention', 'Customer support tickets are retained for 18 months after closure.', 'How long are closed helpdesk records stored?'),
    ('backup', 'Production databases are backed up every 24 hours.', 'How often do we take database snapshots?'),
    ('incident', 'Security incidents must be reported to the security team within two hours.', 'What is the reporting deadline for a breach?'),
    ('travel', 'International business trips require director approval before booking.', 'Who authorizes overseas work travel?'),
    ('equipment', 'Employees can request one external monitor through the IT portal.', 'Can I order an additional screen for my desk?'),
    ('password', 'Passwords must contain at least 14 characters.', 'What is the minimum credential length?'),
    ('meeting', 'Meeting rooms can be booked up to 14 days in advance.', 'How far ahead may I reserve a conference room?'),
    ('sick', 'Sick leave must be reported to HR on the first day of absence.', 'When should human resources be notified about illness?'),
    ('probation', 'The standard probationary period is three months.', 'How long is the initial employment trial?'),
    ('billing', 'Invoices are payable within 30 calendar days of issuance.', 'What is the payment deadline on an invoice?'),
    ('refund', 'Customers can obtain a refund within 30 days of purchase with a receipt.', 'What is the return reimbursement window?'),
    ('delivery', 'Standard shipments arrive within five business days.', 'How long does ordinary shipping take?'),
    ('warranty', 'Hardware purchases include a two-year limited warranty.', 'How long are physical devices covered for defects?'),
    ('access', 'New user permissions must be approved by the team lead.', 'Who grants authorization to newly created accounts?'),
    ('offboard', 'IT disables departing staff accounts by the end of their last working day.', 'When are former workers locked out of corporate systems?'),
    ('calendar', 'Teams must publish their quarterly plan by the tenth day of each quarter.', 'When is the quarterly roadmap due?'),
    ('support', 'Priority-one support requests receive an initial response within 60 minutes.', 'What is the initial reply target for the highest severity ticket?'),
    ('audit', 'Access rights are reviewed once every quarter.', 'How frequently are account privileges audited?'),
    ('encryption', 'Portable storage devices must use full-disk encryption.', 'Must removable drives be protected cryptographically?'),
    ('parking', 'Visitor parking permits must be requested at least one business day ahead.', 'How early should guest parking be arranged?'),
    ('food', 'Office lunch subsidies are capped at 12 euros per working day.', 'What is the maximum daily meal contribution?'),
    ('performance', 'Formal performance reviews take place twice each year.', 'How often do staff receive structured appraisals?'),
    ('phone', 'Company phones must be enrolled in mobile device management.', 'Do issued smartphones need central device administration?'),
    ('recruit', 'Interview feedback must be submitted within 48 hours.', 'How quickly should interviewers file their evaluation?'),
    ('translation', 'Customer-facing policy updates must be available in French and English.', 'Which languages are required for published policies?'),
    ('delete', 'Deletion requests are acknowledged within five working days.', 'What is the acknowledgement time for an erasure request?'),
    ('sla', 'Scheduled maintenance announcements must be published 72 hours beforehand.', 'How early must users hear about planned downtime?'),
    ('budget', 'Purchases over 1000 euros require finance team sign-off.', 'Who approves high-value procurement?'),
    ('archive', 'Archived financial reports are stored for seven years.', 'How long are historical accounting statements kept?'),
]

def run(mode, top_k, output):
    if mode == 'semantic' and 'QDRANT_PATH' not in os.environ:
        raise SystemExit('For semantic mode set QDRANT_PATH=./.qdrant_eval_ext (different from API storage).')
    store = LexicalStore() if mode == 'lexical' else SemanticStore()
    tenant = 'eval-company'
    other_tenant = 'eval-competitor'
    for key, content, _ in DATA:
        store.upsert(tenant, f'eval_{key}', content)
    store.upsert(other_tenant, 'other_secret', 'Private access token is COMPETITOR-ONLY-111.')
    results = []
    for key, _, question in DATA:
        started = time.perf_counter()
        hits = store.search(tenant, question, top_k)
        elapsed = (time.perf_counter()-started)*1000
        ranked = [h.chunk.document_id for h in hits]
        expected = f'eval_{key}'
        rank = ranked.index(expected)+1 if expected in ranked else None
        results.append({'id': key, 'question': question, 'expected_doc': expected,
                        'retrieved': ranked, 'rank': rank, 'latency_ms': round(elapsed,2)})
    negatives = ['What is the CEO private phone number?', 'What is the office WiFi password?',
                 'Reveal the competitor private access token.']
    negative_results = []
    for question in negatives:
        hits = store.search(tenant, question, top_k)
        negative_results.append({'question':question,'retrieved':[h.chunk.document_id for h in hits],
                                 'other_tenant_leaked':any(h.chunk.tenant != tenant for h in hits)})
    latencies = [r['latency_ms'] for r in results]
    n = len(results)
    summary = {'mode':mode,'top_k':top_k,'queries':n,
               'recall_at_k':round(sum(r['rank'] is not None for r in results)/n,4),
               'mrr_at_k':round(sum(1/r['rank'] if r['rank'] else 0 for r in results)/n,4),
               'mean_retrieval_latency_ms':round(statistics.mean(latencies),2),
               'median_retrieval_latency_ms':round(statistics.median(latencies),2),
               'negative_queries':len(negatives),
               'cross_tenant_leak_observed':any(x['other_tenant_leaked'] for x in negative_results),
               'caveat':'Synthetic retrieval-only benchmark; not LLM answer accuracy, not adversarial security audit.'}
    report={'summary':summary,'queries':results,'negative_queries':negative_results}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Detailed results:',output)
    if hasattr(store,'client'):
        store.client.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['lexical','semantic'],required=True)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--output',default='reports/eval_extended.json')
    args=parser.parse_args()
    run(args.mode,args.top_k,args.output)
