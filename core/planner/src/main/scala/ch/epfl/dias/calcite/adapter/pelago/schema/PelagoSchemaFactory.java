package ch.epfl.dias.calcite.adapter.pelago.schema;

import org.apache.calcite.model.ModelHandler;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaFactory;
import org.apache.calcite.schema.SchemaPlus;

import java.io.File;
import java.util.Map;

/**
 * Factory that creates a {@link PelagoSchema}.
 *
 * <p>Allows a custom schema to be included in a <code><i>model</i>.json</code>
 * file.
 *
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvSchemaFactory.java
 */
@SuppressWarnings("UnusedDeclaration")
public class PelagoSchemaFactory implements SchemaFactory {
  /** Public singleton, per factory contract. */
  public static final PelagoSchemaFactory INSTANCE = new PelagoSchemaFactory();

  private PelagoSchemaFactory() {
  }

  public Schema create(SchemaPlus parentSchema, String name,
      Map<String, Object> operand) {
    final String directory = (String) operand.get("directory");
    final File base =
        (File) operand.get(ModelHandler.ExtraOperand.BASE_DIRECTORY.camelName);
    File directoryFile = new File(directory);
    if (base != null && !directoryFile.isAbsolute()) {
      directoryFile = new File(base, directory);
    }
    return new PelagoSchema(directoryFile);
  }
}

// End PelagoSchemaFactory.java
