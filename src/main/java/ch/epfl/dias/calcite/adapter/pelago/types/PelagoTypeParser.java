package ch.epfl.dias.calcite.adapter.pelago.types;

import org.apache.calcite.linq4j.tree.Primitive;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class PelagoTypeParser {

    public static RelDataType parseType(RelDataTypeFactory typeFactory, Map<String, ?> type) throws IOException {
        switch ((String) type.getOrDefault("type", null)){
            case "int":
                return parseInt(typeFactory, type);
            case "int64":
                return parseInt64(typeFactory, type);
            case "float":
                return parseFloat(typeFactory, type);
            case "bool":
                return parseBoolean(typeFactory, type);
            case "dstring":
                return parseDString(typeFactory, type);
            case "date":
                return parseDate(typeFactory, type);
            case "string":
                return parseString(typeFactory, type);
            case "set":
                return parseSet(typeFactory, type);
            case "bag":
                return parseBag(typeFactory, type);
            case "list":
                return parseList(typeFactory, type);
            case "record":
                return parseRecord(typeFactory, type);
            default:
                throw new IOException("unknown type: " + type);
        }
    }

    public static RelDataType parseInt(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("int"));
        RelDataType javaType = typeFactory.createJavaType(Primitive.INT.boxClass);
        RelDataType sqlType = typeFactory.createSqlType(javaType.getSqlTypeName());
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseInt64(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("int64"));
        RelDataType javaType = typeFactory.createJavaType(Primitive.LONG.boxClass);
        RelDataType sqlType = typeFactory.createSqlType(javaType.getSqlTypeName());
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseFloat(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("float"));
        RelDataType javaType = typeFactory.createJavaType(Primitive.DOUBLE.boxClass);
        RelDataType sqlType = typeFactory.createSqlType(javaType.getSqlTypeName());
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseDate(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("date"));
        RelDataType sqlType = typeFactory.createSqlType(SqlTypeName.DATE);
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseBoolean(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("bool"));
        RelDataType javaType = typeFactory.createJavaType(Primitive.BOOLEAN.boxClass);
        RelDataType sqlType = typeFactory.createSqlType(javaType.getSqlTypeName());
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseDString(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("dstring"));
        RelDataType javaType = typeFactory.createJavaType(String.class);
        RelDataType sqlType = typeFactory.createSqlType(SqlTypeName.VARCHAR);
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseString(RelDataTypeFactory typeFactory, Map<String, ?> type){
        assert(type.getOrDefault("type", null).equals("string"));
        RelDataType javaType = typeFactory.createJavaType(String.class);
        RelDataType sqlType = typeFactory.createSqlType(SqlTypeName.VARCHAR);
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseSet(RelDataTypeFactory typeFactory, Map<String, ?> type) throws IOException {
        assert(type.getOrDefault("type", null).equals("set"));
        RelDataType innerType = parseType(typeFactory, (Map<String, ?>) type.get("inner"));
        RelDataType sqlType = typeFactory.createMultisetType(innerType, -1); //TODO: not really a multiset
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseBag(RelDataTypeFactory typeFactory, Map<String, ?> type) throws IOException {
        assert(type.getOrDefault("type", null).equals("bag"));
        RelDataType innerType = parseType(typeFactory, (Map<String, ?>) type.get("inner"));
        RelDataType sqlType = typeFactory.createMultisetType(innerType, -1);
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseList(RelDataTypeFactory typeFactory, Map<String, ?> type) throws IOException {
        assert(type.getOrDefault("type", null).equals("list"));
        RelDataType innerType = parseType(typeFactory, (Map<String, ?>) type.get("inner"));
        RelDataType sqlType = typeFactory.createArrayType(innerType, -1);
        return typeFactory.createTypeWithNullability(sqlType, true);
    }

    public static RelDataType parseRecord(RelDataTypeFactory typeFactory, Map<String, ?> type) throws IOException {
        assert(type.getOrDefault("type", null).equals("record"));
        List<Map<String, ?>> attributes = (List<Map<String, ?>>) type.get("attributes");
        List<RelDataType> types = new ArrayList<RelDataType>();
        List<String     > names = new ArrayList<String     >();
        for (Map<String, ?> attr: attributes){
            types.add(parseType(typeFactory, (Map<String, ?>) attr.get("type")));
            names.add((String) attr.get("attrName"));
        }
        return typeFactory.createStructType(Pair.zip(names, types));
    }
}
