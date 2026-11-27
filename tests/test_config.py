import re
import pytest


class TestConstants:
    
    def test_generic_words_not_empty(self, GENERIC_WORDS):
        assert len(GENERIC_WORDS) > 0
        assert "pepper" in GENERIC_WORDS
        assert "cheese" in GENERIC_WORDS

    def test_staples_not_empty(self, STAPLES):
        assert len(STAPLES) > 0
        assert "salt" in STAPLES
        assert "black pepper" in STAPLES

    def test_fractions_defined(self, FRACTIONS):
        assert "¼" in FRACTIONS
        assert "½" in FRACTIONS
        assert "¾" in FRACTIONS

    def test_units_pattern_compiles(self, UNITS):
        # Should not raise
        re.compile(UNITS)

    def test_preparation_pattern_compiles(self, PREPARATION):
        re.compile(PREPARATION)

    def test_pasta_names_not_empty(self, PASTA_NAMES):
        assert "spaghetti" in PASTA_NAMES
        assert "penne" in PASTA_NAMES

    def test_aliases_not_empty(self, ALIASES):
        assert ALIASES["parmesan cheese"] == "parmesan"
        assert ALIASES["mozzarella cheese"] == "mozzarella"

    def test_data_typos_not_empty(self, DATA_TYPOS):
        assert DATA_TYPOS["armesan"] == "parmesan"

    def test_no_singular_not_empty(self, NO_SINGULAR):
        assert "asparagus" in NO_SINGULAR
        assert "gnocchi" in NO_SINGULAR

    def test_salt_pepper_regex_compiles(self, SALT_PEPPER):
        assert SALT_PEPPER.fullmatch("salt and pepper")
        assert SALT_PEPPER.fullmatch("pepper and salt")
        assert SALT_PEPPER.fullmatch("salt and black pepper")


class TestIngredientKeys:
    
    def test_basic(self):
        from config import ingredient_keys
        keys = ingredient_keys("yellow onion")
        assert "yellow onion" in keys
        assert "onion" in keys

    def test_filters_generic_words(self):
        from config import ingredient_keys, GENERIC_WORDS
        keys = ingredient_keys("black pepper")
        # "pepper" is generic, should be filtered out
        # "black pepper" is not generic, should be kept
        assert "pepper" not in keys
        assert "black pepper" in keys
        assert "pepper" in GENERIC_WORDS

    def test_single_word(self):
        from config import ingredient_keys
        keys = ingredient_keys("pasta")
        assert keys == {"pasta"}

    def test_multi_word(self):
        from config import ingredient_keys
        keys = ingredient_keys("fresh basil leaves")
        assert "fresh basil leaves" in keys
        assert "basil leaves" in keys
        assert "leaves" in keys


class TestToFeatures:

    def test_mode_ingredient(self):
        from config import to_features
        items = ["spaghetti", "tomatoes", "garlic"]
        result = to_features(items, mode="ingredient", drop_staples=False)
        assert result == items

    def test_mode_word(self):
        from config import to_features
        items = ["olive oil", "black pepper"]
        result = to_features(items, mode="word", drop_staples=False)
        assert "w_olive" in result
        assert "w_oil" in result
        assert "w_black" in result
        assert "w_pepper" in result

    def test_mode_both(self):
        from config import to_features
        items = ["spaghetti", "olive oil"]
        result = to_features(items, mode="both", drop_staples=False)
        assert "spaghetti" in result
        assert "olive oil" in result
        assert "w_spaghetti" in result
        assert "w_olive" in result
        assert "w_oil" in result

    def test_drop_staples(self):
        from config import to_features, STAPLES
        items = ["spaghetti", "salt", "black pepper"]
        result = to_features(items, mode="ingredient", drop_staples=True)
        assert "salt" not in result
        assert "black pepper" not in result
        assert "spaghetti" in result


class TestCleanIngredientsV2:

    def test_basic_cleaning(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["2 cloves garlic, minced"])
        assert "garlic" in result

    def test_fraction_removal(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["½ cup olive oil"])
        assert "olive oil" in result
        assert "½" not in str(result)

    def test_unit_removal(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["1 pound spaghetti"])
        # spaghetti gets normalized to "pasta" due to PASTA_NAMES
        assert "pasta" in result
        assert "pound" not in str(result)

    def test_preparation_removal(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["chopped tomatoes"])
        # tomatoes gets singularized to "tomato"
        assert "tomato" in result
        assert "chopped" not in str(result)

    def test_alias_normalization(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["parmesan cheese"])
        assert "parmesan" in result
        assert "cheese" not in str(result)

    def test_typo_correction(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["armesan"])
        assert "parmesan" in result

    def test_salt_and_pepper_split(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["salt and pepper"])
        assert "salt" in result
        assert "black pepper" in result

    def test_pasta_normalization(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["spaghetti"])
        assert "pasta" in result

    def test_deduplication(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["garlic", "garlic cloves", "cloves garlic"])
        # All should normalize to "garlic" and be deduplicated
        assert result.count("garlic") == 1

    def test_unicode_handling(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["jalapeño"])
        assert "jalapeno" in result

    def test_parentheses_removal(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["tomatoes (8 ounce)"])
        # tomatoes gets singularized to "tomato"
        assert "tomato" in result
        assert "ounce" not in str(result)

    def test_or_handling(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["butter or margarine"])
        assert "butter" in result
        assert "margarine" not in str(result)

    def test_leaves_singularization(self):
        from config import clean_ingredients_v2
        result = clean_ingredients_v2(["basil leaves"])
        assert "basil" in result
        assert "leaves" not in str(result)


class TestDatasetSHA256:

    def test_sha256_defined(self):
        from config import dataset_sha256
        assert dataset_sha256 is not None
        assert len(dataset_sha256) == 64  # SHA256 hex length

    def test_alias_exists(self):
        from config import dataset_sha256
        assert dataset_sha256 is not None