#!/bin/bash
set -e
for n in 2 3 4 5 6 7 8 9 10 11; do
  echo "===== Trying $n replicas ====="
  docker compose -f docker-compose-replicas.yml down -v 2>/dev/null || true
  docker compose -f docker-compose-replicas.yml up -d --scale api=$n
  sleep 70
  nvidia-smi
  echo ""
  echo "Logs tail:"
  docker compose -f docker-compose-replicas.yml logs --tail 20
  echo ""
  # OOM 체크
  if docker compose -f docker-compose-replicas.yml ps | grep -q "Exit"; then
    echo "OOM at $n replicas - max is $((n-1))"
    break
  fi
done
