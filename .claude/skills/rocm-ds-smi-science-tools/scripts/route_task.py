import sys

def route_task(task_description):
    task_description = task_description.lower()
    if any(kw in task_description for kw in ['pandas', 'dataframe', 'tabular', 'etl']):
        return 'hipdf-pandas-port'
    elif any(kw in task_description for kw in ['vector search', 'ann', 'embedding', 'knn']):
        return 'hipvs-ann'
    elif any(kw in task_description for kw in ['graph', 'pagerank', 'community detection', 'bfs']):
        return 'hipgraph-analytics'
    elif any(kw in task_description for kw in ['clustering', 'linear algebra', 'primitive', 'math']):
        return 'hipraft-primitives'
    elif any(kw in task_description for kw in ['oom', 'memory', 'fragmentation', 'pool']):
        return 'hipmm-memory-ops'
    else:
        return 'compatibility-triage' # Default safe fallback

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f"Routed to: {route_task(sys.argv[1])}")
    else:
        print("Please provide a task description.")
