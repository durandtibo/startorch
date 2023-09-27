import torch
from coola import objects_are_equal
from pytest import mark, raises

from startorch import constants as ct
from startorch.example.hypercube import create_hypercube_examples
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###############################################
#     Tests for create_hypercube_examples     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_create_hypercube_examples_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        create_hypercube_examples(num_examples=num_examples)


@mark.parametrize("num_classes", (0, -1))
def test_create_hypercube_examples_incorrect_num_classes(num_classes: int) -> None:
    with raises(RuntimeError, match="he number of classes .* has to be greater than 0"):
        create_hypercube_examples(num_classes=num_classes)


def test_create_hypercube_examples_incorrect_feature_size() -> None:
    with raises(
        RuntimeError,
        match="The feature dimension .* has to be greater or equal to the number of classes .*",
    ):
        create_hypercube_examples(num_classes=50, feature_size=32)


def test_create_hypercube_examples_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        create_hypercube_examples(noise_std=-1)


def test_create_hypercube_examples_create() -> None:
    data = create_hypercube_examples(num_examples=10, num_classes=5, feature_size=8)
    assert len(data) == 2
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.long
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_create_hypercube_examples_create_num_examples(num_examples: int) -> None:
    data = create_hypercube_examples(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("num_classes", SIZES)
def test_create_hypercube_examples_create_num_classes(num_classes: int) -> None:
    targets = create_hypercube_examples(num_examples=10, num_classes=num_classes)[ct.TARGET]
    assert targets.min() >= 0
    assert targets.max() < num_classes


@mark.parametrize("feature_size", SIZES)
def test_create_hypercube_examples_create_feature_size(feature_size: int) -> None:
    data = create_hypercube_examples(num_examples=10, num_classes=1, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


def test_create_hypercube_examples_noise_std_0() -> None:
    features = create_hypercube_examples(num_examples=10, noise_std=0)[ct.FEATURE]
    assert features.min() == 0
    assert features.max() == 1


def test_create_hypercube_examples_create_same_random_seed() -> None:
    assert objects_are_equal(
        create_hypercube_examples(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
        create_hypercube_examples(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
    )


def test_create_hypercube_examples_create_different_random_seeds() -> None:
    assert not objects_are_equal(
        create_hypercube_examples(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
        create_hypercube_examples(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(2)
        ),
    )
