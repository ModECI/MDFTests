set -e

python model_with_simple_graph.py
python model_with_nested_graph.py &
python model_ABCD.py &

python SimpleFN.py

echo "Tested all examples"

