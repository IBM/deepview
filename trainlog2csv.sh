awk '
BEGIN {
  tagid = 0
}
/START/ {
  tag = $8
  getline
  cmd = $0
  # printf("%s, \"%s\"\n", tag, cmd)
  tagid_tbl[tagid] = tag
  index_tbl[tag] = 0
  tagid += 1
}
/^Loss before bwd:/ {
  loss = $4
  idx = index_tbl[tag]
  loss_tbl[tag, idx] = loss
  index_tbl[tag] += 1
}
$7 ~ /avg_loss/ {
  loss = substr($6, 2, length($6) - 3)
  printf("%d, %s\n", $2, loss)
  idx = index_tbl[tag]
  loss_tbl[tag, idx] = loss
  index_tbl[tag] += 1
}
END {
  max_idx = 0
  for (id = 0; id < tagid; id++) {
    tag = tagid_tbl[id]
    printf("%s, ", tag)
    max_idx = (index_tbl[tag] > max_idx) ? index_tbl[tag] : max_idx
  }
  printf("\n")
  for (idx = 0; idx < max_idx; idx++) {
    for (id = 0; id < tagid; id++) {
      tag = tagid_tbl[id]
      printf("%s, ", loss_tbl[tag, idx])
    }
    printf("\n")
  }
}
'
