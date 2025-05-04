def convert_dms_to_decimal(value: float) -> float:
    """
    緯度経度の60進法を10進法に変換する
    """
    integer_part = int(value)
    decimal_part_dms = value - integer_part
    decimal_part_dec = decimal_part_dms * 100 / 60

    return integer_part + decimal_part_dec