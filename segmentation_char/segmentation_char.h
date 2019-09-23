//
// Created by niko on 9/13/19.
//

#ifndef CODE_SEGMENTATION_CHAR_H
#define CODE_SEGMENTATION_CHAR_H

#include <cstdint>
#include <iostream>

union Segmentation {
  static_assert(sizeof(double) == 8,
      "Assumption of IEEE floating point necessiates 8 bytes per double");
  double d;
  unsigned char c[sizeof(double)];
  uint64_t u;
};

template <int start, int end>
struct Extract_helper {
  static_assert(start >= 0, "Extracting bytes before start of double");
  static_assert(end < sizeof(double), "Extracting bytes beyond end of double");
  static_assert(start < end, "Only forward copying is supported");

  static constexpr void copy(double d, unsigned char* c) {
    Segmentation seg{d};
    *c = seg.c[sizeof(double) - start - 1];
    Extract_helper<start + 1, end> eh;
    eh.copy(d, c + 1);
  }
};

template <int index>
struct Extract_helper<index, index> {
  static_assert(index >= 0, "Extracting bytes before end of double");
  static_assert(index < sizeof(double), "Extracting bytes beyond end of double");

  static constexpr void copy(double d, unsigned char* c) {
    Segmentation seg{d};
    *c = seg.c[sizeof(double) - index - 1];
  }
};

template <int start, int end>
void extract_slice(double d, unsigned char (&chars)[end - start + 1]) {
  static_assert(start >= 0, "Extracting bytes before start of double");
  static_assert(end < sizeof(double), "Extracting bytes beyond end of double");
  static_assert(start <= end, "Extracting negative sized double slice");

  Extract_helper<start, end> eh;
  eh.copy(d, chars);
}

template <int start, int end>
struct Insert_helper {
  static_assert(start >= 0, "Inserting bytes before start of double");
  static_assert(end < sizeof(double), "Inserting bytes beyond end of double");
  static_assert(start < end, "Inserting negative sized slice");

  static constexpr uint64_t insert(const unsigned char* bytes) {
    Segmentation seg{0.};
    seg.c[sizeof(double) - start - 1] = *bytes;
    Insert_helper<start + 1, end> ih;
    return seg.u | ih.insert(bytes + 1);
  }
};

template <int index>
struct Insert_helper<index, index> {
  static_assert(index >= 0, "Inserting bytes before start of double");
  static_assert(index < sizeof(double), "Inserting bytes beyond end of double");

  static constexpr uint64_t insert(const unsigned char* bytes) {
    Segmentation seg{0.};
    seg.c[sizeof(double) - index - 1] = *bytes;
    return seg.u;
  }
};

template <int start, int end>
double insert_slice(const unsigned char (&chars)[end - start + 1]) {
  static_assert(start >= 0, "Inserting bytes before start of double");
  static_assert(end < sizeof(double), "Inserting bytes beyond end of double");
  static_assert(start <= end, "Inserting negative sized or empty slice");

  Insert_helper<start, end> ih;
  uint64_t u = ih.insert(chars);
  Segmentation s{0.};
  s.u = u;
  return s.d;
}

// TODO: implement special case to fill/ insert whole double for double_slice
template <int start, int end>
struct Double_slice {
  static_assert(start >= 0, "Slice started before start of double");
  static_assert(end < sizeof(double), "Slice exceeding end of double");

  explicit Double_slice() : Double_slice(0.) {}

  explicit Double_slice(double d) {
    extract_slice<start, end>(d, bytes);
  }

  double to_double() const {
    return insert_slice<start, end>(bytes);
  }

  bool operator==(const Double_slice<start, end>& other) const {
    // TODO: make more efficient
    bool eq = true;
    for (int i = 0; i < end - start + 1; ++i) {
      eq = eq && bytes[i] == other.bytes[i];
    }
    return eq;
  }

  void print_bytes() const {
    for (auto b : bytes) {
      std::cout << (+b & 0xFF) << " ";
    }
    std::cout << std::endl;
  }

private:
  unsigned char bytes[end - start + 1];
};

template <>
struct Double_slice<0, sizeof(double) - 1> {
public:
  explicit Double_slice() : Double_slice(0.) {}

  explicit Double_slice(double d) : d{d} {}

  double to_double() const {
    return d;
  }

  bool operator==(const Double_slice<0, sizeof(double) - 1>& other) const {
    return d == other.d;
  }

  void print_bytes() const {
    Segmentation seg{d};
    for (auto c : seg.c) {
      std::cout << (+c & 0xFF) << " ";
    }
    std::cout << std::endl;
  }

private:
  double d;
};

#endif // CODE_SEGMENTATION_CHAR_H
