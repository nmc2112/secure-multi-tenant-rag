"""Run retrieval baseline vs semantic retrieval and report Recall@k and latency."""
import argparse, json, statistics, time
from app.core import LexicalStore, SemanticStore

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['lexical','semantic'],default='lexical')
p.add_argument('--top-k',type=int,default=3)
args=p.parse_args()
store=LexicalStore() if args.mode=='lexical' else SemanticStore()
rows=[json.loads(l) for l in open('data/eval.jsonl')]
for row in rows: store.upsert(row['tenant'],row['document_id'],row['content'])
latencies=[]; hits=0
for row in rows:
    start=time.perf_counter(); res=store.search(row['tenant'],row['question'],args.top_k)
    latencies.append((time.perf_counter()-start)*1000)
    hits+=any(h.chunk.document_id==row['expected_doc'] for h in res)
print(json.dumps({'mode':args.mode,'queries':len(rows),'recall_at_k':hits/len(rows),'mean_retrieval_latency_ms':round(statistics.mean(latencies),2),'top_k':args.top_k},indent=2))
