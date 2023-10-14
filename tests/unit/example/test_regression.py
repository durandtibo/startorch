import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import make_normal_linear_regression
from startorch.example.regression import get_uniform_weights
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###################################################
#     Tests for make_normal_linear_regression     #
###################################################


@mark.parametrize("num_examples", (0, -1))
def test_make_normal_linear_regression_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_normal_linear_regression(num_examples=num_examples)


@mark.parametrize("feature_size", (0, -1))
def test_make_normal_linear_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(RuntimeError, match="feature_size (.*) has to be greater than 0"):
        make_normal_linear_regression(feature_size=feature_size)


def test_make_normal_linear_regression_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_normal_linear_regression(noise_std=-1)


def test_make_normal_linear_regression() -> None:
    data = make_normal_linear_regression(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].batch_size == 10
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].batch_size == 10
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


def test_make_normal_linear_regression_weights() -> None:
    data = make_normal_linear_regression(num_examples=10, feature_size=8, weights=torch.ones(8))
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].batch_size == 10
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].batch_size == 10
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


def test_make_normal_linear_regression_incorrect_weights() -> None:
    with raises(RuntimeError, match=r"shape '\[8, 1\]' is invalid for input of size"):
        make_normal_linear_regression(num_examples=10, feature_size=8, weights=torch.ones(10))


@mark.parametrize("num_examples", SIZES)
def test_make_normal_linear_regression_num_examples(num_examples: int) -> None:
    data = make_normal_linear_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("feature_size", SIZES)
def test_make_normal_linear_regression_feature_size(feature_size: int) -> None:
    data = make_normal_linear_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape == (10, feature_size)


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_normal_linear_regression_create_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_normal_linear_regression(
            num_examples=64, noise_std=noise_std, feature_size=8, generator=get_torch_generator(1)
        ),
        make_normal_linear_regression(
            num_examples=64, noise_std=noise_std, feature_size=8, generator=get_torch_generator(1)
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_normal_linear_regression_create_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_normal_linear_regression(
            num_examples=64, noise_std=noise_std, feature_size=8, generator=get_torch_generator(1)
        ),
        make_normal_linear_regression(
            num_examples=64, noise_std=noise_std, feature_size=8, generator=get_torch_generator(2)
        ),
    )


#########################################
#     Tests for get_uniform_weights     #
#########################################


def test_get_uniform_weights_informative_feature_size_0() -> None:
    assert get_uniform_weights(feature_size=10, informative_feature_size=0).equal(
        torch.zeros(10, 1)
    )


def test_get_uniform_weights_informative_feature_size_10() -> None:
    assert not get_uniform_weights(feature_size=10, informative_feature_size=10).equal(
        torch.zeros(10, 1)
    )


@mark.parametrize("informative_feature_size", (0, 1, 5, 10))
def test_get_uniform_weights_same_random_seed(informative_feature_size: int) -> None:
    assert objects_are_equal(
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("informative_feature_size", (1, 5, 10))
def test_get_uniform_weights_different_random_seeds(informative_feature_size: int) -> None:
    assert not objects_are_equal(
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(2),
        ),
    )
