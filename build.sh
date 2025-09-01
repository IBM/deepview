
. ./env.sh

(
$SEN_PROJECT_SRC/aiu-toolbox/examples/openshift/dev-image/bin/rebuild-llvm.sh
$SEN_PROJECT_SRC/aiu-toolbox/examples/openshift/dev-image/bin/rebuild-stack.sh
)  2>&1 | tee ${0%.*}.log
