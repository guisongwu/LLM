# from tokenizers.pre_tokenizers import ByteLevel
# alphabet = ByteLevel.alphabet()
# print(' ' in alphabet)

from tokenizers.pre_tokenizers import ByteLevel

alphabet = ByteLevel.alphabet()
print(len(alphabet))       # 256
print(ord(' '))            # 32
print(alphabet[32])        # 看看 byte 32 映射成哪个 Unicode
