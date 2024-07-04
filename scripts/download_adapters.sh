tasks=(
    math
    code
    intent
    summary
    law
    translation
)

mkdir -p all_models/adapters/token
git lfs install
for i in {0..5}
do
    git clone https://huggingface.co/deepseek-ai/ESFT-token-${tasks[$i]}-lite ./all_models/adapters/token/${tasks[$i]}
done


mkdir -p all_models/adapters/gate
for i in {0..5}
do
    git clone https://huggingface.co/deepseek-ai/ESFT-gate-${tasks[$i]}-lite ./all_models/adapters/gate/${tasks[$i]}
done
