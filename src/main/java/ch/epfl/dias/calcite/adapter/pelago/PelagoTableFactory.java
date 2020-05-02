package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.model.ModelHandler;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.TableFactory;
import org.apache.calcite.util.Source;
import org.apache.calcite.util.Sources;

import java.io.File;
import java.util.Map;

/**
 * Factory that creates a {@link PelagoTable}.
 *
 * <p>Allows a table to be included in a model.json file, even in a
 * schema that is not based upon {@link PelagoSchema}.
 *
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableFactory.java
 */
@SuppressWarnings("UnusedDeclaration")
public class PelagoTableFactory implements TableFactory<PelagoTable> {
    // public constructor, per factory contract
    public PelagoTableFactory() {
    }

    public PelagoTable create(SchemaPlus schema, String name,
                           Map<String, Object> operand, RelDataType rowType) {
        String fileName = (String) operand.get("file");
        final File base =
                (File) operand.get(ModelHandler.ExtraOperand.BASE_DIRECTORY.camelName);
        final Source source = Sources.file(base, fileName);
        final RelProtoDataType protoRowType =
                rowType != null ? RelDataTypeImpl.proto(rowType) : null;

        try {
            return PelagoTable.create(source, name, (Map<String, ?>) operand.get("plugin"), protoRowType);
        } catch (MalformedPlugin malformedPlugin) {
            malformedPlugin.printStackTrace();
            return null;
        }
    }

}

// End CsvTableFactory.java
