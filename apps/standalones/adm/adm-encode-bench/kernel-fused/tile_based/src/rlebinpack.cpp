#include "ssb_utils.h"
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>

using namespace std;

pair<uint, uint> rleBinPack(uint*&in, uint*& value, uint*& run_length, uint*& val_offsets, uint*& rl_offsets, uint num_entries) {
  uint val_offset = 0;
  uint rl_offset = 0;

  uint block_size = 512;
  uint elem_per_thread = 1;
  uint tile_size = block_size * elem_per_thread;

  //nonblock
  block_size = tile_size;

  uint miniblock_count = 4;
  uint total_count = num_entries;
  uint first_val = in[0];

  value[0] = block_size;
  value[1] = miniblock_count;
  value[2] = total_count;
  value[3] = first_val;

  run_length[0] = block_size;
  run_length[1] = miniblock_count;
  run_length[2] = total_count;
  run_length[3] = first_val;

  val_offset += 4;
  rl_offset += 4;

  // uint num_tiles = (num_entries + tile_size - 1) / tile_size;

  uint* val = new uint[tile_size]();
  uint* rl = new uint[tile_size]();

  for (uint tile_start=0; tile_start<num_entries; tile_start += tile_size) {
    uint block_index = tile_start / block_size;

    uint count = 0;
    val[count] = in[0];
    uint run = 1;
    for (int i = 1; i < tile_size; i++) {
      if (in[i] != in[i-1]) {
        rl[count] = run;
        count++;
        val[count] = in[i];
        run = 1;
      } else {
        run++;
      }
    }
    rl[count] = run;
    count++;

    int bl_size = count;
    int block_start = 0;

    rl_offsets[block_index] = rl_offset;
    val_offsets[block_index] = val_offset;

    uint min_val = val[block_start];
    uint min_rl = rl[block_start];
    for (int i = 1; i < bl_size; i++) {
      if (val[block_start + i] < min_val) min_val = val[block_start + i];
      if (rl[block_start + i] < min_rl) min_rl = rl[block_start + i];
    }

    uint val_bitwidth = 0;
    uint rl_bitwidth = 0;

    for (int i = block_start; i < block_start + bl_size; i++) {
      val[i] = val[i] - min_val;
      rl[i] = rl[i] - min_rl;
      uint bitwidth = uint(ceil(log2(val[i] + 1)));
      val_bitwidth = max(val_bitwidth, bitwidth);
      bitwidth = uint(ceil(log2(rl[i] + 1)));
      rl_bitwidth = max(rl_bitwidth, bitwidth);
    }

    value[val_offset] = min_val;
    run_length[rl_offset] = min_rl;
    val_offset++; rl_offset++;

    value[val_offset] = val_bitwidth + (val_bitwidth << 8) +
      (val_bitwidth << 16) + (val_bitwidth << 24);
    run_length[rl_offset] = rl_bitwidth + (rl_bitwidth << 8) +
      (rl_bitwidth << 16) + (rl_bitwidth << 24);
    // cout << value[val_offset] << " " << run_length[val_offset] << endl;
    val_offset++; rl_offset++;

    if (block_start == (bl_size * (elem_per_thread - 1))) { // if last block
      value[val_offset] = count - bl_size * (elem_per_thread - 1);
      run_length[rl_offset] = count - bl_size * (elem_per_thread - 1);
    } else {
      value[val_offset] = bl_size;
      run_length[rl_offset] = bl_size;
    }
    val_offset++; rl_offset++;

    uint bitwidth = val_bitwidth;
    uint shift = 0;
    for (int i = block_start; i < block_start + bl_size; i++) {
      if (shift + bitwidth > 32) {
        if (shift != 32) value[val_offset] += val[i] << shift;
        val_offset++;
        shift = (shift + bitwidth) & (32-1);
        value[val_offset] = val[i] >> (bitwidth - shift);
      } else {
        value[val_offset] += val[i] << shift;
        shift += bitwidth;
      }
    }
    val_offset++;

    bitwidth = rl_bitwidth;
    shift = 0;
    for (int i = block_start; i < block_start + bl_size; i++) {
      if (shift + bitwidth > 32) {
        if (shift != 32) run_length[rl_offset] += rl[i] << shift;
        rl_offset++;
        shift = (shift + bitwidth) & (32-1);
        run_length[rl_offset] = rl[i] >> (bitwidth - shift);
      } else {
        run_length[rl_offset] += rl[i] << shift;
        shift += bitwidth;
      }
    }
    rl_offset++;

    in += tile_size;

  }

  val_offsets[num_entries / block_size] = val_offset;
  rl_offsets[num_entries / block_size] = rl_offset;

  cout << val_offset << " " << rl_offset << endl;

  return make_pair(val_offset, rl_offset);
}

int storeEncodedValueColumn(string col_name, uint* value, uint arr_byte_size, uint* val_offsets, uint num_blocks) {
  string filename = DATA_DIR + lookup(col_name) + ".valbin";
  ofstream colData (filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    cout << "Unable to write column" << endl;
    return -1;
  }

  colData.write((char*)value, arr_byte_size);
  colData.close();

  string offsets_filename = DATA_DIR + lookup(col_name) + ".valbinoff";
  ofstream offsetsData (offsets_filename.c_str(), ios::out | ios::binary);
  if (!offsetsData) {
    cout << "Unable to write offsets" << endl;
    return -1;
  }

  offsetsData.write((char*)val_offsets, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return 0;
}

int storeEncodedRunLengthColumn(string col_name, uint* run_length, uint arr_byte_size, uint* rl_offsets, uint num_blocks) {
  string filename = DATA_DIR + lookup(col_name) + ".rlbin";
  ofstream colData (filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    cout << "Unable to write column" << endl;
    return -1;
  }

  colData.write((char*)run_length, arr_byte_size);
  colData.close();

  string offsets_filename = DATA_DIR + lookup(col_name) + ".rlbinoff";
  ofstream offsetsData (offsets_filename.c_str(), ios::out | ios::binary);
  if (!offsetsData) {
    cout << "Unable to write offsets" << endl;
    return -1;
  }

  offsetsData.write((char*)rl_offsets, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return 0;
}

int main(int argc, char** argv) {
  // if (argc != 2) {
  //   cout << "encode <col-name>" << endl;
  //   return 1;
  // }

  string col_name = "lo_orderkey";
  int len = LO_LEN;

  uint *raw = loadColumn<uint>(col_name, len);

  cout << "Loaded Column" << endl;

  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint *col = new uint[adjusted_len]();
  memcpy(col, raw, len * sizeof(uint));

  uint *value = new uint[adjusted_len]();
  uint *run_length = new uint[adjusted_len]();
  uint *val_offsets = new uint[num_blocks + 1]();
  uint *rl_offsets = new uint[num_blocks + 1]();

  // extend with the last value to make it multiple of 128
  for (int i = len; i < adjusted_len ;i++) col[i] = raw[len-1];

  pair<uint, uint> ret = rleBinPack(col, value, run_length, val_offsets, rl_offsets, adjusted_len);
  cout << "Num Elements " << len << endl;
  cout << "Input: ArrSize " << len * 4 << endl;
  cout << "Output: ArrSize " << (ret.first + ret.second) * 4 << " Offsets " << num_blocks + 1 << endl;

  storeEncodedValueColumn(col_name, value, ret.first * 4, val_offsets, num_blocks);
  storeEncodedRunLengthColumn(col_name, run_length, ret.second * 4, rl_offsets, num_blocks);

  return 0;
}
