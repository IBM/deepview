. env.sh

(
# Make the $SEN_PROJECT_SRC
mkdir -p $SEN_PROJECT_SRC

# Save $SEN_PROJECT_SRC/build/llvm
rm -rf $SEN_PROJECT_SRC/llvm.save
mv $SEN_PROJECT_SRC/build/llvm $SEN_PROJECT_SRC/llvm.save
# Checkout the sourc code
./checkout-src.sh $SEN_PROJECT_SRC
# Restore $SEN_PROJECT_SRC/build/llvm
mkdir -p $SEN_PROJECT_SRC/build
mv $SEN_PROJECT_SRC/llvm.save $SEN_PROJECT_SRC/build/llvm

# Checkout other repositories
git clone https://huggingface.co/google-bert/bert-base-uncased $SEN_PROJECT_SRC/bert-base-uncased

) 2>&1 | tee ${0%.*}.log
