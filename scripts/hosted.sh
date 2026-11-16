#!/bin/bash
# Search-distillation rounds on hosted machines: CPU collection and evaluation jobs, GPU
# training jobs, and the two halves of a round. The job bodies are
# docs/briefs/15-artifacts/hosted_cpu_job.py and gpu_train_job.py; this file fills them in,
# submits them, waits and downloads. Every step is skipped when its result exists, so a
# command that was interrupted is simply run again.
#
# Settings, from the environment (never commit them: the repository is public):
#   FOG_HOSTED_USER     account name on the host
#   FOG_HOSTED_DATASET  name of the private dataset that holds rounds and weights
#   FOG_HOSTED_WORK     a scratch folder outside the repository (job folders, downloads, markers)
#   FOG_HOSTED_CLI      the host's command-line client (default: kaggle)
#
# Commands:
#   hosted.sh cpu-job <letter> <timeout_s> '<STEPS json>'   submit job fog-distill-collect-<letter>
#   hosted.sh fetch <letter> <tag> [target]                  wait, download, unpack into target
#   hosted.sh gpu-train <tag> '<RUNS json>'                  submit, wait, download to work/kout-<tag>
#   hosted.sh upload <message> [file ...]                    copy files into the dataset, new version
#   hosted.sh round-collect <round> <seconds> [letters]      one round's collection, here and hosted
#   hosted.sh round-train <round> ['<extra RUNS entries>']   train on the GPU, then play here
#
# Run round-collect and round-train from a job runner that keeps the session alive; each can
# take up to two hours. Collected rounds live in runs/distill/round<N>[<letter>], networks in
# runs/distill/net<N>, play results in runs/distill/eval/.
set -u
: "${FOG_HOSTED_USER:?set FOG_HOSTED_USER}" "${FOG_HOSTED_DATASET:?set FOG_HOSTED_DATASET}"
: "${FOG_HOSTED_WORK:?set FOG_HOSTED_WORK}"
K=${FOG_HOSTED_CLI:-kaggle}; U=$FOG_HOSTED_USER; W=$FOG_HOSTED_WORK
R=$(cd "$(dirname "$0")/.." && pwd); PY=$R/.venv/bin/python
mkdir -p "$W/markers" "$W/dataset"
hide() { sed "s/$U/<user>/g"; }
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$W/loop.log"; }
commit() { git -C "$R" rev-parse origin/work/roadmap; }

fill() {  # fill <template> <variable> <value json> <folder> <slug> <gpu true|false>
  "$PY" - "$@" "$(commit)" "$U" "$FOG_HOSTED_DATASET" <<'E'
import json, re, sys
template, variable, value, folder, slug, gpu, commit, user, dataset = sys.argv[1:]
source = open(template).read()
source, n = re.subn(r'^COMMIT = .*$', f'COMMIT = "{commit}"', source, flags=re.M)
assert n == 1
pattern = rf'^{variable} = \[.*?\]$'
source, n = re.subn(pattern, lambda _: f'{variable} = ' + repr(json.loads(value)), source,
                    flags=re.M | re.S)
assert n == 1
compile(source, 'job.py', 'exec')
open(f'{folder}/job.py', 'w').write(source)
json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": gpu == "true",
           "enable_internet": True, "dataset_sources": [f"{user}/{dataset}"],
           "competition_sources": [], "kernel_sources": []},
          open(f'{folder}/kernel-metadata.json', 'w'), indent=1)
E
}

wait_for() {  # wait_for <slug>: print the final status line
  local status
  while true; do
    status=$($K kernels status "$U/$1" 2>&1 | hide)
    case "$status" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
    sleep 45
  done
  echo "$status"
}

cpu_job() {
  local letter=$1 timeout=$2 steps=$3 folder=$W/cpu-$1
  mkdir -p "$folder"
  fill "$R/docs/briefs/15-artifacts/hosted_cpu_job.py" STEPS "$steps" "$folder" \
    "fog-distill-collect-$letter" false || return 1
  $K kernels push -p "$folder" --timeout "$timeout" 2>&1 | hide | tail -1
}

fetch() {
  local letter=$1 tag=$2 target=${3:-$R/runs/distill} slug=fog-distill-collect-$1
  wait_for "$slug"
  mkdir -p "$W/kout-$tag" "$target"
  $K kernels output "$U/$slug" -p "$W/kout-$tag" > "$W/kout-$tag.log" 2>&1
  cat "$W/kout-$tag/job.json" 2>/dev/null
  for archive in "$W/kout-$tag"/*.tar; do
    [ -f "$archive" ] && tar -xf "$archive" -C "$target" && echo "unpacked $(basename "$archive")"
  done
  [ -f "$W/kout-$tag/job.json" ] || tail -30 "$W/kout-$tag/$slug.log" 2>/dev/null | cut -c1-400
}

gpu_train() {
  local tag=$1 runs=$2 folder=$W/gpu slug=fog-distill-train-gpu
  mkdir -p "$folder"
  if [ ! -f "$W/markers/gpu-$tag.pushed" ]; then
    fill "$R/docs/briefs/15-artifacts/gpu_train_job.py" RUNS "$runs" "$folder" "$slug" true || return 1
    $K quota --format json > "$W/quota-before-$tag.json" 2>/dev/null
    $K kernels push -p "$folder" --accelerator NvidiaTeslaT4 --timeout 3600 2>&1 | hide | tail -1
    touch "$W/markers/gpu-$tag.pushed"
    sleep 20
  fi
  wait_for "$slug"
  mkdir -p "$W/kout-$tag"
  $K kernels output "$U/$slug" -p "$W/kout-$tag" > "$W/kout-$tag.log" 2>&1
  $K quota --format json > "$W/quota-after-$tag.json" 2>/dev/null
  cat "$W/kout-$tag/job.json" 2>/dev/null
}

upload() {  # upload <message> [file ...]
  local message=$1; shift
  [ -f "$W/dataset/dataset-metadata.json" ] || { echo "put dataset-metadata.json in $W/dataset"; return 1; }
  for file in "$@"; do cp "$file" "$W/dataset/"; done
  $K datasets version -p "$W/dataset" -m "$message" 2>&1 | hide | tail -1
  sleep 20
  until $K datasets status "$U/$FOG_HOSTED_DATASET" 2>&1 | grep -qi ready; do sleep 15; done
}

collect_step() {  # collect_step <folder name> <seed> <network file> <seconds>
  cat <<E
[{"name":"$1","command":["-m","fog.training.search_distill","collect","--pool","data/teams/replay_v1/train","--out","{out}/$1","--checkpoint","{input:$3}","--games","4000","--beta","0.25","--seed","$2","--agreement-every","40","--workers","2","--max-seconds","$4"]}]
E
}

round_collect() {
  local round=$1 seconds=$2 letters=${3:-"a b d e"} previous=$(($1 - 1)) index=0 letter name
  cd "$R" || return 1
  if [ ! -f "$W/markers/r$round.net_uploaded" ]; then
    cp "runs/distill/net$previous/policy_best.pt" "$W/dataset/net$previous.pt"
    upload "network $previous weights" && touch "$W/markers/r$round.net_uploaded"
    say "round $round: network $previous uploaded"
  fi
  for letter in $letters; do          # hosted machine <letter> writes round<N><letter>
    index=$((index + 1)); name=round$round$letter
    if [ ! -f "$W/markers/r$round.pushed_$letter" ]; then
      cpu_job "$letter" $((seconds + 3000)) \
        "$(collect_step "$name" $(((15 + index) * 1000 + round)) "net$previous.pt" "$seconds")" \
        && touch "$W/markers/r$round.pushed_$letter"
      say "round $round: hosted machine $letter submitted"
    fi
  done
  if [ ! -f "$W/markers/r$round.local_done" ]; then
    say "round $round: local collection for $seconds s"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -m fog.training.search_distill \
      collect --pool data/teams/replay_v1/train --out "runs/distill/round$round" \
      --checkpoint "runs/distill/net$previous/policy_best.pt" --games 4000 --beta 0.25 \
      --seed $((15000 + round)) --agreement-every 40 --workers 4 --max-seconds "$seconds" \
      > "$W/round$round.log" 2>&1 || { say "round $round: local collection failed"; return 1; }
    touch "$W/markers/r$round.local_done"
  fi
  local archives=()
  for letter in "" $letters; do
    name=round$round$letter
    if [ -n "$letter" ] && [ ! -f "runs/distill/$name/summary.json" ]; then
      fetch "$letter" "$name" > "$W/fetch-$name.log" 2>&1
      say "round $round: hosted machine $letter: $(head -1 "$W/fetch-$name.log")"
    fi
    [ -f "runs/distill/$name/summary.json" ] || continue
    tar -cf "$W/$name.tar" --exclude='*.log' --exclude='*.lock' -C runs/distill "$name"
    archives+=("$W/$name.tar")
    "$PY" - "$name" <<'E' | tee -a "$W/loop.log"
import json, sys
s = json.load(open(f"runs/distill/{sys.argv[1]}/summary.json"))
print(f"  {sys.argv[1]}: games {s['games']} rows {s['rows']} "
      f"check {s.get('top1_agreement_with_check_budget', 0):.3f} "
      f"repeat {s.get('top1_agreement_with_second_seed', 0):.3f} "
      f"simulations/label {s.get('teacher_visits_per_search', 0):.0f} "
      f"core-s/row {s['seconds'] / max(1, s['rows']):.3f}")
E
  done
  if [ ! -f "$W/markers/r$round.rounds_uploaded" ]; then
    upload "round $round" "${archives[@]}" && touch "$W/markers/r$round.rounds_uploaded"
  fi
  say "round $round: collection finished"
}

play() {  # play <label> <checkpoint>: the network alone, 1,008 games an opponent
  [ -f "runs/distill/eval/$1-1008/merged.json" ] && return 0
  say "play $1"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -m fog.eval.run_shards \
    --shards 4 --games 1008 --seed 15200 --out-dir "runs/distill/eval/$1-1008" -- \
    --pool data/teams/replay_v1/validation --device cpu --config configs/search_distill_cpu.yaml \
    --checkpoint "$1=$2" --opponent simpleheuristics --opponent maxdamage --max-turns 300 \
    --progress-every 40 > "$W/eval-$1-1008.log" 2>&1
  grep -E '^\| 20' "$W/eval-$1-1008.log" | cut -c1-112 | tee -a "$W/loop.log"
}

round_train() {
  local round=$1 extra=${2:-} net=net$1 folder name
  cd "$R" || return 1
  if [ ! -f "runs/distill/$net/policy_best.pt" ]; then
    say "$net: GPU job"
    gpu_train "$net" "[{\"name\":\"net\",\"rounds\":[],\"args\":[]}$extra]" > "$W/gpu-$net.log" 2>&1
    [ -f "$W/kout-$net/net/policy_best.pt" ] || { say "$net: no checkpoint; see $W/gpu-$net.log"; return 1; }
    for folder in "$W/kout-$net"/*/; do
      name=$(basename "$folder"); [ "$name" = net ] && name=$net
      mkdir -p "runs/distill/$name" && cp "$folder"* "runs/distill/$name/"
    done
    say "$net: trained"
  fi
  for folder in "$W/kout-$net"/*/ "runs/distill/$net/"; do
    name=$(basename "$folder"); [ "$name" = net ] && name=$net
    [ -f "runs/distill/$name/policy_best.pt" ] && play "$name" "runs/distill/$name/policy_best.pt"
  done
  say "$net: done"
}

command=${1:?command}; shift
case "$command" in
  cpu-job) cpu_job "$@" ;;
  fetch) fetch "$@" ;;
  gpu-train) gpu_train "$@" ;;
  upload) upload "$@" ;;
  round-collect) round_collect "$@" ;;
  round-train) round_train "$@" ;;
  play) cd "$R" && play "$@" ;;
  *) echo "unknown command: $command"; exit 2 ;;
esac
