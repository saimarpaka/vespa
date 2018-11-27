package com.yahoo.abicheck.collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.objectweb.asm.Opcodes;

public class Util {

  public static final List<AccessFlag> classFlags = Arrays.asList(
      AccessFlag.make(Opcodes.ACC_PUBLIC, "public"),
      AccessFlag.make(Opcodes.ACC_PRIVATE, "private"),
      AccessFlag.make(Opcodes.ACC_PROTECTED, "protected"),
      AccessFlag.make(Opcodes.ACC_FINAL, "final"),
      AccessFlag.make(Opcodes.ACC_SUPER, null), // Ignored, always set by modern Java
      AccessFlag.make(Opcodes.ACC_INTERFACE, "interface"),
      AccessFlag.make(Opcodes.ACC_ABSTRACT, "abstract"),
      AccessFlag.make(Opcodes.ACC_SYNTHETIC, "synthetic"), // FIXME: Do we want this?
      AccessFlag.make(Opcodes.ACC_ANNOTATION, "annotation"),
      AccessFlag.make(Opcodes.ACC_ENUM, "enum")
// FIXME: Module support
//      AccessFlag.make(Opcodes.ACC_MODULE, "module")
  );

  public static final List<AccessFlag> methodFlags = Arrays.asList(
      AccessFlag.make(Opcodes.ACC_PUBLIC, "public"),
      AccessFlag.make(Opcodes.ACC_PRIVATE, "private"),
      AccessFlag.make(Opcodes.ACC_PROTECTED, "protected"),
      AccessFlag.make(Opcodes.ACC_STATIC, "static"),
      AccessFlag.make(Opcodes.ACC_FINAL, "final"),
      AccessFlag.make(Opcodes.ACC_SYNCHRONIZED, "synchronized"),
      AccessFlag.make(Opcodes.ACC_BRIDGE, "bridge"),
      AccessFlag.make(Opcodes.ACC_VARARGS, "varargs"), // FIXME: Do we want this?
      AccessFlag.make(Opcodes.ACC_NATIVE, "native"),
      AccessFlag.make(Opcodes.ACC_ABSTRACT, "abstract"),
      AccessFlag.make(Opcodes.ACC_STRICT, "strict"), // FIXME: Do we want this?
      AccessFlag.make(Opcodes.ACC_SYNTHETIC, "synthetic") // FIXME: Do we want this?
  );

  public static List<String> convertAccess(int access, List<AccessFlag> flags) {
    List<String> result = new ArrayList<>();
    for (AccessFlag flag : flags) {
      if ((access & flag.bit) != 0 && flag.attribute != null) {
        result.add(flag.attribute);
      }
      access &= ~flag.bit;
    }
    if (access != 0) {
      throw new IllegalArgumentException(String.format("Unexpected access bits: 0x%x", access));
    }
    return result;
  }

  private static class AccessFlag {

    public final int bit;
    public final String attribute;

    private AccessFlag(int bit, String attribute) {
      this.bit = bit;
      this.attribute = attribute;
    }

    private static AccessFlag make(int bit, String attribute) {
      return new AccessFlag(bit, attribute);
    }
  }
}
