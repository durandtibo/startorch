from __future__ import annotations

import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import (
    Acosh,
    Asinh,
    Atanh,
    Cosh,
    SincTensorTransformer,
    Sinh,
    Tanh,
)
from startorch.utils.seed import get_torch_generator

###########################
#     Tests for Acosh     #
###########################


def test_acosh_str() -> None:
    assert str(Acosh()).startswith("AcoshTensorTransformer(")


def test_acosh_transform() -> None:
    assert objects_are_allclose(
        Acosh().transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [0.0, 1.316957950592041, 1.7627471685409546],
                [2.063436985015869, 2.292431592941284, 2.477888822555542],
            ]
        ),
    )


def test_acosh_transform_same_random_seed() -> None:
    transformer = Acosh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_acosh_transform_different_random_seeds() -> None:
    transformer = Acosh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################
#     Tests for Asinh     #
###########################


def test_asinh_str() -> None:
    assert str(Asinh()).startswith("AsinhTensorTransformer(")


def test_asinh_transform() -> None:
    assert objects_are_allclose(
        Asinh().transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [0.8813735842704773, 1.4436354637145996, 1.8184465169906616],
                [2.094712495803833, 2.3124382495880127, 2.4917798042297363],
            ]
        ),
    )


def test_asinh_transform_same_random_seed() -> None:
    transformer = Asinh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_asinh_transform_different_random_seeds() -> None:
    transformer = Asinh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################
#     Tests for Atanh     #
###########################


def test_atanh_str() -> None:
    assert str(Atanh()).startswith("AtanhTensorTransformer(")


def test_atanh_transform() -> None:
    assert objects_are_allclose(
        Atanh().transform(torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]])),
        torch.tensor(
            [
                [-0.5493061542510986, -0.10033535212278366, 0.0],
                [0.10033535212278366, 0.20273256301879883, 0.5493061542510986],
            ]
        ),
    )


def test_atanh_transform_same_random_seed() -> None:
    transformer = Atanh()
    tensor = torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_atanh_transform_different_random_seeds() -> None:
    transformer = Atanh()
    tensor = torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################
#     Tests for Cosh     #
##########################


def test_cosh_str() -> None:
    assert str(Cosh()).startswith("CoshTensorTransformer(")


def test_cosh_transform() -> None:
    assert objects_are_allclose(
        Cosh().transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [1.5430806875228882, 3.762195587158203, 10.067662239074707],
                [27.3082332611084, 74.20994567871094, 201.71563720703125],
            ]
        ),
    )


def test_cosh_transform_same_random_seed() -> None:
    transformer = Cosh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_cosh_transform_different_random_seeds() -> None:
    transformer = Cosh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for SincTensorTransformer     #
###########################################


def test_sinc_str() -> None:
    assert str(SincTensorTransformer()).startswith("SincTensorTransformer(")


def test_sinc_transform() -> None:
    assert objects_are_allclose(
        SincTensorTransformer().transform(torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])),
        torch.tensor(
            [
                [1.0, 0.9836316704750061, 0.9354892373085022],
                [0.858393669128418, 0.7568266987800598, 0.6366197466850281],
            ]
        ),
    )


def test_sinc_transform_same_random_seed() -> None:
    transformer = SincTensorTransformer()
    tensor = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_sinc_transform_different_random_seeds() -> None:
    transformer = SincTensorTransformer()
    tensor = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################
#     Tests for Sinh     #
##########################


def test_sinh_str() -> None:
    assert str(Sinh()).startswith("SinhTensorTransformer(")


def test_sinh_transform() -> None:
    assert objects_are_allclose(
        Sinh().transform(torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [0.0, 1.175201177597046, 3.6268603801727295],
                [27.2899169921875, 74.20320892333984, 201.71315002441406],
            ]
        ),
    )


def test_sinh_transform_same_random_seed() -> None:
    transformer = Sinh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_sinh_transform_different_random_seeds() -> None:
    transformer = Sinh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################
#     Tests for Tanh     #
##########################


def test_tanh_str() -> None:
    assert str(Tanh()).startswith("TanhTensorTransformer(")


def test_tanh_transform() -> None:
    assert objects_are_allclose(
        Tanh().transform(torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [0.0, 0.7615941762924194, 0.9640275835990906],
                [0.9993293285369873, 0.9999092221260071, 0.9999877214431763],
            ]
        ),
    )


def test_tanh_transform_same_random_seed() -> None:
    transformer = Tanh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_tanh_transform_different_random_seeds() -> None:
    transformer = Tanh()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
