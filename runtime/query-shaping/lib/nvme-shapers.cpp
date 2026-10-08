/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include <query-shaping/nvme-shapers.hpp>

using namespace proteus;

std::vector<DanglingAttr> CPUOnlyNVMeMorsel::constructDanglingAttrs(
    const std::string &relName,
    const std::vector<std::string> &relAttrs) const {
  std::vector<DanglingAttr> attrs;
  CatalogParser catalog = CatalogParser(catalog_path);
  auto inputInfo = catalog.getInputInfo(catalog_path + "/" + relName + ".csv");
  auto &collType = dynamic_cast<CollectionType &>(*(inputInfo->exprType));

  const ExpressionType &nestedType = collType.getNestedType();
  auto record_type = dynamic_cast<const RecordType &>(nestedType);

  for (const auto &attr : relAttrs) {
    auto arg = record_type.getArg(attr);
    if (!arg) {
      LOG(FATAL) << "Attribute " << attr << " not found";
    }
    switch (arg->getOriginalType()->getTypeID()) {
      case BOOL:
        attrs.push_back(dangling_attr::Bool(attr));
        break;
      case DSTRING:
        attrs.push_back(dangling_attr::DString(attr));
        break;
      case STRING:
        attrs.push_back(dangling_attr::String(attr));
        break;
      case FLOAT:
        attrs.push_back(dangling_attr::Float(attr));
        break;
      case INT:
        attrs.push_back(dangling_attr::Int(attr));
        break;
      case DATE:
        attrs.push_back(dangling_attr::Date(attr));
        break;
      case INT64:
        attrs.push_back(dangling_attr::Int64(attr));
        break;
      case RECORD:
      case LIST:
      case BAG:
      case SET:
      case COMPOSITE:
      case BLOCK:
      case INDEXEDSEQ:
        LOG(FATAL) << "Unsupported type";
    }
  }
  return attrs;
}

std::vector<std::filesystem::path> CPUOnlyNVMeMorsel::getMdForAttribute(
    const std::string &relName, const std::string &attr) const {
  std::vector<std::filesystem::path> md_paths;
  //     for each input dir, find all associated metadata files for this
  //     attribute of this relation.
  // Anchor the match to "<relName>.csv.<attr>_" at the START of the file
  // basename. Plain substring matching over-matches when an attribute name is
  // shared across relations (Stack's bare FK columns, e.g. site_id appears in
  // many tables), when one attribute name contains another (id vs site_id), or
  // when one relation name is a suffix of another (question vs tag_question).
  const std::string prefix = relName + ".csv." + attr + "_";
  for (const auto &dir : input_dirs) {
    auto dir_files = getSortedDirectoryFiles(dir);
    for (const auto &entry : dir_files) {
      const std::string name = entry.filename().string();
      if (name.rfind(prefix, 0) == 0 &&
          name.find("metadata.json") != std::string::npos) {
        md_paths.push_back(entry);
      }
    }
  }
  std::sort(md_paths.begin(), md_paths.end());
  return md_paths;
}

RelBuilder CPUOnlyNVMeMorsel::scan(
    const std::string &relName, std::initializer_list<std::string> relAttrs) {
  std::vector<DanglingAttr> attrs = constructDanglingAttrs(relName, relAttrs);
  RecordType scan_record_type = rel(relName)(attrs);

  auto meta_data_records_map = scan_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
      relation_md;

  for (const auto &attr : relAttrs) {
    auto md_paths = getMdForAttribute(relName, attr);
    relation_md.emplace_back(meta_data_records_map[attr], md_paths);
  }
  auto builder = getBuilder();
  auto rel = builder.scan(relation_md);
  rel = rel.hintRowCount(getRowHint(relName));
  return rel;
}
