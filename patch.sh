
#DRYRUN=--dry-run

. env.sh

(
patch -p 1 $DRYRUN -d $SEN_PROJECT_SRC/deeptools < ./issue3499.patch
patch -p 1 $DRYRUN -d $SEN_PROJECT_SRC/deeptools < ./issue3633.patch
patch -p 1 $DRYRUN -d $SEN_PROJECT_SRC/deeptools < ./pr3721.patch
#patch -p 1 $DRYRUN -d $SEN_PROJECT_SRC/torch_sendnn < ./pr262.patch
patch -p 1 $DRYRUN -d $SEN_PROJECT_SRC/torch_sendnn < ./backends_fix.patch
) 2>&1 | tee ${0%.*}.log
