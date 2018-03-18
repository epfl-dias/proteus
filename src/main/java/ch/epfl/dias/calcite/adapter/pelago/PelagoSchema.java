package ch.epfl.dias.calcite.adapter.pelago;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.impl.AbstractSchema;
import org.apache.calcite.util.Source;
import org.apache.calcite.util.Sources;

import com.google.common.collect.ImmutableMap;

import java.io.File;
import java.nio.file.Paths;
import java.util.Map;

/**
 * Schema from a Pelago catalog.
 */
public class PelagoSchema extends AbstractSchema {
  private final File directoryFile;
  private Map<String, Table> tableMap;

  /**
   * Creates a Pelago schema.
   *
   * @param directoryFile Directory that holds catalog.json file
   */
  public PelagoSchema(File directoryFile) {
    super();
    this.directoryFile = directoryFile;
  }

  @Override protected Map<String, Table> getTableMap() {
    if (tableMap == null) tableMap = createTableMap();
    return tableMap;
  }

  private Map<String, Table> createTableMap() {
    // Build a map from table name to table; each file becomes a table.
    final ImmutableMap.Builder<String, Table> builder = ImmutableMap.builder();
    ObjectMapper mapper = new ObjectMapper();
    Map<String, ?> catalog;
    try {
        catalog = mapper.readValue(Paths.get(directoryFile.toPath().toString(), "catalog.json").toFile(), new TypeReference<Map<String, ?>>() {});
    } catch (java.io.IOException e){
        System.err.println("Catalog not found " + directoryFile + " not found");
        return builder.build();
    }

    for (Map.Entry<String, ?> e: catalog.entrySet()) {
        System.out.println("Table Found: " + e.getKey());
        System.out.println("   Row Type: " + ((Map<String, ?>) e.getValue()).get("type").toString());
        Map<String, ?> fileEntry = (Map<String, ?>) ((Map<String, ?>) e.getValue()).get("type");
        String fileType = (String) fileEntry.getOrDefault("type", null);
        if (!fileType.equals("bag")) {
            System.err.println("Error in catalog: relation type is expected to be \"bag\", but \"" + fileType + "\" found");
            System.out.println("Ignoring table: " + e.getKey());
            continue;
        }
        Map<String, ?> lineType = (Map<String, ?>) fileEntry.getOrDefault("inner", null);
        if (lineType != null && !lineType.getOrDefault("type", null).equals("record")) lineType = null;
        if (lineType == null) {
            System.err.println("Error in catalog: \"bag\" expected to contain records");
            System.out.println("Ignoring table: " + e.getKey());
            continue;
        }
        Source source = Sources.of(new File((String) ((Map<String, ?>) e.getValue()).get("path")));

        Map<String, ?> plugin = (Map<String, ?>) ((Map<String, ?>) e.getValue()).getOrDefault("plugin", null);
        if (plugin == null) {
            System.err.println("Error in catalog: plugin information not found for table");
            System.out.println("Ignoring table: " + e.getKey());
            continue;
        }

        Object obj_linehint = plugin.getOrDefault("lines",  null);
        if (obj_linehint == null){
            obj_linehint = plugin.getOrDefault("linehint",  null);
        }
        Long linehint = null;
        if (obj_linehint != null) {
            if (obj_linehint instanceof Integer) {
                linehint = ((Integer) obj_linehint).longValue();
            } else if (obj_linehint instanceof Long){
                linehint = (Long) obj_linehint;
            } else {
                System.err.println("Error in catalog: unrecognized type for \"lines\"");
                System.out.println("Ignoring table: " + e.getKey());
                continue;
            }
        }

        if (linehint == null) {
            System.err.println("Error in catalog: \"lines\" not found for table");
            System.out.println("Ignoring table: " + e.getKey());
            continue;
        }

        final Table table = new PelagoTable(source, lineType, plugin, linehint);
        builder.put(e.getKey(), table); //.toUpperCase(Locale.getDefault())
    }
    return builder.build();
  }
}

// End PelagoSchema.java
