Build image with:

docker build -t vgg16_fashion_mnist .

Run image to test it on fashion-mnist testset and print results with:

docker run -it --rm vgg16_fashion_mnist python load_and_test_model.py