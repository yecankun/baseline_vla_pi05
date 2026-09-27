#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_PROJECT="/home/zsw/project_2026"
SOURCE_HF_CACHE="/home/zsw/.cache/huggingface/hub/models--lerobot--pi05_base"
SOURCE_FUTURE_MODELS="/home/zsw/models/project_2026"
TARGET_MOUNT="/media/zsw/SSD1T"
TARGET_ROOT="${TARGET_MOUNT}/project_2026_weights_v1"
TARGET_PROJECT_TREE="${TARGET_ROOT}/project_tree"
TARGET_HF_CACHE="${TARGET_ROOT}/huggingface/hub/models--lerobot--pi05_base"
TARGET_FUTURE_MODELS="${TARGET_ROOT}/models"
STATE_DIR="${SOURCE_PROJECT}/simulation_output/weight_migration_to_ssd1t_v1"
PATH_MANIFEST="${STATE_DIR}/source_weight_paths.txt"
RESULT_TSV="${STATE_DIR}/migration_results.tsv"
SUMMARY_FILE="${STATE_DIR}/summary.txt"
MIN_FREE_BUFFER_BYTES=$((10 * 1024 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  bash tools/migrate_project_weights_to_ssd1t.sh --plan
  bash tools/migrate_project_weights_to_ssd1t.sh --execute

--plan is read-only. --execute copies each weight, verifies SHA256, replaces
the original file/cache directory with a compatibility symlink, and only then
removes the verified source-side backup.
EOF
}

fail() {
  echo "[error] $*" >&2
  exit 1
}

require_layout() {
  [[ "$(id -un)" == "zsw" ]] || fail "expected user zsw"
  [[ -d "${SOURCE_PROJECT}" ]] || fail "missing source project: ${SOURCE_PROJECT}"
  [[ -d "${TARGET_MOUNT}" ]] || fail "missing target mount: ${TARGET_MOUNT}"
  mountpoint -q "${TARGET_MOUNT}" || fail "target is not a mount point: ${TARGET_MOUNT}"
  [[ "$(findmnt -n -o FSTYPE --target "${TARGET_MOUNT}")" == "ext4" ]] ||
    fail "target mount is not ext4"
  [[ -w "${TARGET_MOUNT}" ]] || fail "target mount is not writable"
  [[ "$(stat -c %d "${SOURCE_PROJECT}")" != "$(stat -c %d "${TARGET_MOUNT}")" ]] ||
    fail "source and target unexpectedly share one filesystem"
  [[ "${TARGET_ROOT}" == "${TARGET_MOUNT}/project_2026_weights_v1" ]] ||
    fail "unexpected target root"
}

list_source_weights() {
  find "${SOURCE_PROJECT}" -xdev -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \
       -o -name '*.safetensors' -o -name '*.bin' -o -name '*.onnx' \) \
    -print | LC_ALL=C sort
}

project_bytes_from_paths() {
  local total=0 path
  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    total=$((total + $(stat -Lc %s -- "${path}")))
  done
  echo "${total}"
}

hf_bytes() {
  if [[ -L "${SOURCE_HF_CACHE}" ]]; then
    du -sbL "${SOURCE_HF_CACHE}" | awk '{print $1}'
  elif [[ -d "${SOURCE_HF_CACHE}" ]]; then
    du -sb "${SOURCE_HF_CACHE}" | awk '{print $1}'
  else
    fail "missing PI0.5 cache: ${SOURCE_HF_CACHE}"
  fi
}

plan() {
  local paths project_count project_bytes cache_bytes total_bytes free_bytes
  paths="$(list_source_weights)"
  project_count="$(printf '%s\n' "${paths}" | sed '/^$/d' | wc -l)"
  project_bytes="$(printf '%s\n' "${paths}" | project_bytes_from_paths)"
  cache_bytes="$(hf_bytes)"
  total_bytes=$((project_bytes + cache_bytes))
  free_bytes="$(df --output=avail -B1 "${TARGET_MOUNT}" | tail -1 | tr -d ' ')"

  echo "mode=plan"
  echo "source_project=${SOURCE_PROJECT}"
  echo "target_root=${TARGET_ROOT}"
  echo "project_weight_count=${project_count}"
  echo "project_weight_bytes=${project_bytes}"
  echo "pi05_cache_bytes=${cache_bytes}"
  echo "total_required_bytes=${total_bytes}"
  echo "target_free_bytes=${free_bytes}"
  echo "required_free_with_buffer_bytes=$((total_bytes + MIN_FREE_BUFFER_BYTES))"
  [[ "${free_bytes}" -ge $((total_bytes + MIN_FREE_BUFFER_BYTES)) ]] ||
    fail "target lacks required free space plus 10 GiB safety buffer"
  printf '%s\n' "${paths}"
}

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

record_result() {
  local status="$1" bytes="$2" sha="$3" source="$4" target="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${bytes}" "${sha}" "${source}" "${target}" >> "${RESULT_TSV}"
}

migrate_one_file() {
  local source="$1" relative target partial backup temp_link
  local source_bytes source_sha target_sha

  [[ "${source}" == "${SOURCE_PROJECT}/"* ]] || fail "source escaped project root: ${source}"
  relative="${source#${SOURCE_PROJECT}/}"
  target="${TARGET_PROJECT_TREE}/${relative}"
  partial="${target}.partial"
  backup="${source}.project2026-migration-backup"
  temp_link="${source}.project2026-ssd-link-tmp"

  [[ "${target}" == "${TARGET_PROJECT_TREE}/"* ]] || fail "target escaped project tree"

  if [[ -L "${source}" ]]; then
    [[ "$(readlink -- "${source}")" == "${target}" ]] ||
      fail "unexpected existing symlink target: ${source}"
    source_bytes="$(stat -Lc %s -- "${source}")"
    source_sha="$(sha256_file "${source}")"
    record_result "already_migrated" "${source_bytes}" "${source_sha}" "${source}" "${target}"
    return
  fi

  if [[ ! -e "${source}" && -f "${backup}" ]]; then
    mv -- "${backup}" "${source}"
  fi
  [[ -f "${source}" ]] || fail "missing regular source file: ${source}"
  [[ ! -e "${backup}" ]] || fail "unexpected migration backup exists: ${backup}"

  source_bytes="$(stat -c %s -- "${source}")"
  source_sha="$(sha256_file "${source}")"
  mkdir -p -- "$(dirname -- "${target}")"

  if [[ -f "${target}" ]] && [[ "$(stat -c %s -- "${target}")" == "${source_bytes}" ]] &&
     [[ "$(sha256_file "${target}")" == "${source_sha}" ]]; then
    :
  else
    rm -f -- "${partial}"
    rsync -a --partial -- "${source}" "${partial}"
    [[ "$(stat -c %s -- "${partial}")" == "${source_bytes}" ]] ||
      fail "copied size mismatch: ${source}"
    target_sha="$(sha256_file "${partial}")"
    [[ "${target_sha}" == "${source_sha}" ]] || fail "copied SHA256 mismatch: ${source}"
    mv -f -- "${partial}" "${target}"
  fi

  target_sha="$(sha256_file "${target}")"
  [[ "${target_sha}" == "${source_sha}" ]] || fail "final target SHA256 mismatch: ${source}"

  rm -f -- "${temp_link}"
  ln -s -- "${target}" "${temp_link}"
  mv -- "${source}" "${backup}"
  mv -T -- "${temp_link}" "${source}"
  if [[ "$(sha256_file "${source}")" != "${source_sha}" ]]; then
    rm -f -- "${source}"
    mv -- "${backup}" "${source}"
    fail "compatibility symlink verification failed: ${source}"
  fi
  rm -f -- "${backup}"
  record_result "migrated" "${source_bytes}" "${source_sha}" "${source}" "${target}"
  echo "[migrated] ${source_bytes} ${source}"
}

tree_digest() {
  local root="$1"
  (
    cd "${root}"
    while IFS= read -r relative; do
      printf '%s  %s\n' "$(sha256_file "${relative}")" "${relative}"
    done < <(find . -type f -printf '%P\n' | LC_ALL=C sort)
    while IFS= read -r relative; do
      printf 'SYMLINK  %s  %s\n' "${relative}" "$(readlink -- "${relative}")"
    done < <(find . -type l -printf '%P\n' | LC_ALL=C sort)
  ) | sha256sum | awk '{print $1}'
}

migrate_hf_cache() {
  local partial backup temp_link source_digest target_digest cache_bytes
  partial="${TARGET_HF_CACHE}.partial"
  backup="${SOURCE_HF_CACHE}.project2026-migration-backup"
  temp_link="${SOURCE_HF_CACHE}.project2026-ssd-link-tmp"

  if [[ -L "${SOURCE_HF_CACHE}" ]]; then
    [[ "$(readlink -- "${SOURCE_HF_CACHE}")" == "${TARGET_HF_CACHE}" ]] ||
      fail "unexpected existing HF cache symlink"
    source_digest="$(tree_digest "${SOURCE_HF_CACHE}")"
    record_result "already_migrated_tree" "$(hf_bytes)" "${source_digest}" \
      "${SOURCE_HF_CACHE}" "${TARGET_HF_CACHE}"
    return
  fi

  if [[ ! -e "${SOURCE_HF_CACHE}" && -d "${backup}" ]]; then
    mv -- "${backup}" "${SOURCE_HF_CACHE}"
  fi
  [[ -d "${SOURCE_HF_CACHE}" ]] || fail "missing regular HF cache directory"
  [[ ! -e "${backup}" ]] || fail "unexpected HF migration backup exists"

  source_digest="$(tree_digest "${SOURCE_HF_CACHE}")"
  cache_bytes="$(hf_bytes)"
  mkdir -p -- "$(dirname -- "${TARGET_HF_CACHE}")"

  if [[ -d "${TARGET_HF_CACHE}" ]] &&
     [[ "$(tree_digest "${TARGET_HF_CACHE}")" == "${source_digest}" ]]; then
    :
  else
    mkdir -p -- "${partial}"
    rsync -a --partial -- "${SOURCE_HF_CACHE}/" "${partial}/"
    target_digest="$(tree_digest "${partial}")"
    [[ "${target_digest}" == "${source_digest}" ]] || fail "HF cache tree digest mismatch"
    [[ ! -e "${TARGET_HF_CACHE}" ]] || fail "unverified final HF target exists"
    mv -- "${partial}" "${TARGET_HF_CACHE}"
  fi

  target_digest="$(tree_digest "${TARGET_HF_CACHE}")"
  [[ "${target_digest}" == "${source_digest}" ]] || fail "final HF cache digest mismatch"

  rm -f -- "${temp_link}"
  ln -s -- "${TARGET_HF_CACHE}" "${temp_link}"
  mv -- "${SOURCE_HF_CACHE}" "${backup}"
  mv -T -- "${temp_link}" "${SOURCE_HF_CACHE}"
  if [[ "$(tree_digest "${SOURCE_HF_CACHE}")" != "${source_digest}" ]]; then
    rm -f -- "${SOURCE_HF_CACHE}"
    mv -- "${backup}" "${SOURCE_HF_CACHE}"
    fail "HF compatibility symlink verification failed"
  fi
  [[ "${backup}" == "/home/zsw/.cache/huggingface/hub/models--lerobot--pi05_base.project2026-migration-backup" ]] ||
    fail "unexpected HF backup path"
  rm -rf -- "${backup}"
  record_result "migrated_tree" "${cache_bytes}" "${source_digest}" \
    "${SOURCE_HF_CACHE}" "${TARGET_HF_CACHE}"
  echo "[migrated] ${cache_bytes} ${SOURCE_HF_CACHE}"
}

create_future_model_link() {
  mkdir -p -- "$(dirname -- "${SOURCE_FUTURE_MODELS}")" "${TARGET_FUTURE_MODELS}"
  if [[ -L "${SOURCE_FUTURE_MODELS}" ]]; then
    [[ "$(readlink -- "${SOURCE_FUTURE_MODELS}")" == "${TARGET_FUTURE_MODELS}" ]] ||
      fail "unexpected future-model symlink target"
  elif [[ -e "${SOURCE_FUTURE_MODELS}" ]]; then
    fail "future-model path exists and is not the expected symlink: ${SOURCE_FUTURE_MODELS}"
  else
    ln -s -- "${TARGET_FUTURE_MODELS}" "${SOURCE_FUTURE_MODELS}"
  fi
}

execute_migration() {
  local project_count project_bytes cache_bytes total_bytes free_before free_after migrated_count
  mkdir -p -- "${STATE_DIR}" "${TARGET_ROOT}"
  if [[ ! -f "${PATH_MANIFEST}" ]]; then
    list_source_weights > "${PATH_MANIFEST}"
  fi
  project_count="$(sed '/^$/d' "${PATH_MANIFEST}" | wc -l)"
  project_bytes="$(project_bytes_from_paths < "${PATH_MANIFEST}")"
  cache_bytes="$(hf_bytes)"
  total_bytes=$((project_bytes + cache_bytes))
  free_before="$(df --output=avail -B1 "${TARGET_MOUNT}" | tail -1 | tr -d ' ')"
  [[ "${free_before}" -ge $((total_bytes + MIN_FREE_BUFFER_BYTES)) ]] ||
    fail "target lacks required free space plus 10 GiB safety buffer"

  printf 'status\tbytes\tsha256_or_tree_digest\tsource\ttarget\n' > "${RESULT_TSV}"
  while IFS= read -r source; do
    [[ -n "${source}" ]] || continue
    migrate_one_file "${source}"
  done < "${PATH_MANIFEST}"
  migrate_hf_cache
  create_future_model_link

  migrated_count="$(tail -n +2 "${RESULT_TSV}" | wc -l)"
  [[ "${migrated_count}" == "$((project_count + 1))" ]] ||
    fail "result count mismatch"
  free_after="$(df --output=avail -B1 "${TARGET_MOUNT}" | tail -1 | tr -d ' ')"
  cat > "${SUMMARY_FILE}" <<EOF
status=passed
project_weight_count=${project_count}
project_weight_bytes=${project_bytes}
pi05_cache_bytes=${cache_bytes}
total_source_bytes=${total_bytes}
result_count=${migrated_count}
target_free_bytes_before=${free_before}
target_free_bytes_after=${free_after}
target_root=${TARGET_ROOT}
project_paths_preserved_by_symlink=true
hf_cache_path_preserved_by_symlink=true
future_model_path_redirected=true
EOF
  cat "${SUMMARY_FILE}"
}

main() {
  [[ $# -eq 1 ]] || { usage; exit 2; }
  require_layout
  case "$1" in
    --plan) plan ;;
    --execute) execute_migration ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"
