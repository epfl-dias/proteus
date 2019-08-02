/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ch.epfl.dias.calcite.adapter.pelago;

import au.com.bytecode.opencsv.CSVReader;

import org.apache.calcite.adapter.enumerable.JavaRowFormat;
import org.apache.calcite.adapter.enumerable.PhysType;
import org.apache.calcite.adapter.enumerable.PhysTypeImpl;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.avatica.util.DateTimeUtils;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Source;

import java.io.IOException;
import java.io.Reader;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Enumerator that reads from a CSV file.
 *
 * @param <E> Row type
 */
public class PelagoEnumerator<E> implements Enumerator<E> {
  private final CSVReader reader;
  private final String[] filterValues;
  private final AtomicBoolean cancelFlag;
  private final RowConverter<E> rowConverter;
  private E current;

  public PelagoEnumerator(Source source, AtomicBoolean cancelFlag,
                  List<RelDataTypeField> fieldTypes, int[] fields,
                  JavaTypeFactory typeFactory, boolean mock) {
    //noinspection unchecked
    this(source, cancelFlag, false, typeFactory, null,
        (RowConverter<E>) converter(fieldTypes, fields, mock));
  }

  public PelagoEnumerator(Source source, AtomicBoolean cancelFlag, boolean stream,
      JavaTypeFactory typeFactory,
      String[] filterValues, RowConverter<E> rowConverter) {
    this.cancelFlag = cancelFlag;
    this.rowConverter = rowConverter;
    this.filterValues = filterValues;
    try {
//      if (stream) {
//        this.reader = new CsvStreamReader(source);
//      } else {
        this.reader = openCsv(source);
//      }
//      this.reader.readNext(); // skip header row //we do not have a header!
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static RowConverter<?> converter(List<RelDataTypeField> fieldTypes,
      int[] fields, boolean mock) {
    if (fields.length == 1) {
      final int field = fields[0];
      return new SingleColumnRowConverter(fieldTypes.get(field), field, mock);
    } else {
      return new ArrayRowConverter(fieldTypes, fields, mock);
    }
  }

  public static CSVReader openCsv(Source source) throws IOException {
    final Reader fileReader = source.reader();
    return new CSVReader(fileReader);
  }

  public E current() {
    return current;
  }

  public boolean moveNext() {
    try {
    outer:
      for (;;) {
        if (cancelFlag.get()) {
          return false;
        }
        final String[] strings = reader.readNext();
        if (strings == null) {
//          if (reader instanceof CsvStreamReader) {
//            try {
//              Thread.sleep(CsvStreamReader.DEFAULT_MONITOR_DELAY);
//            } catch (InterruptedException e) {
//              throw new RuntimeException(e);
//            }
//            continue;
//          }
          current = null;
          reader.close();
          return false;
        }
        if (filterValues != null) {
          for (int i = 0; i < strings.length; i++) {
            String filterValue = filterValues[i];
            if (filterValue != null) {
              if (!filterValue.equals(strings[i])) {
                continue outer;
              }
            }
          }
        }
        current = rowConverter.convertRow(strings);
        return true;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void reset() {
    throw new UnsupportedOperationException();
  }

  public void close() {
    try {
      reader.close();
    } catch (IOException e) {
      throw new RuntimeException("Error closing CSV reader", e);
    }
  }

  /** Returns an array of integers {0, ..., n - 1}. */
  public static int[] identityList(int n) {
    int[] integers = new int[n];
    for (int i = 0; i < n; i++) {
      integers[i] = i;
    }
    return integers;
  }

  /** Row converter.
   *
   * @param <E> element type */
  abstract static class RowConverter<E> {
    abstract E convertRow(String[] rows);
    protected final Random  rand = new Random();
    protected final boolean mock;

    boolean isMock(){
      return mock;
    }

    public RowConverter(boolean mock){
      this.mock = mock;
    }
//
//    public RowConverter(){
//      this(false);
//    }

    protected Object convert(RelDataTypeField fieldType, String string) {

      if (fieldType == null) {
        return string;
      }

      if (fieldType.getType().getSqlTypeName() == SqlTypeName.BOOLEAN) {
        if (mock) return rand.nextBoolean();
        if (string.length() == 0) {
          return null;
        }
        return Boolean.parseBoolean(string);
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.INTEGER) {
        if (mock) return (rand.nextInt() % 2);
        if (string.length() == 0) {
          return null;
        }
        return Integer.parseInt(string);
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.BIGINT) {
        if (mock) return rand.nextLong();
        if (string.length() == 0) {
          return null;
        }
        return Long.parseLong(string);
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.FLOAT) {
        if (mock) return rand.nextFloat();
        if (string.length() == 0) {
          return null;
        }
        return Double.parseDouble(string);
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.DOUBLE) {
        if (mock) return rand.nextDouble();
        if (string.length() == 0) {
          return null;
        }
        return Double.parseDouble(string);
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.DECIMAL) {
        if (mock) return rand.nextDouble();
        if (string.length() == 0) {
          return null;
        }
        return new BigDecimal(Double.parseDouble(string));
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.VARCHAR) {
        if (mock) return UUID.randomUUID().toString().replace("-", "");
        return string;
      } else if (fieldType.getType().getSqlTypeName() == SqlTypeName.DATE) {
        if (mock) string = Long.toString(rand.nextLong());
        if (string.length() == 0) {
          return null;
        }
        System.out.println(string);
        return (int) (Long.parseLong(string) / DateTimeUtils.MILLIS_PER_DAY);
//        try {
//          Date date = TIME_FORMAT_DATE.parse(string);
//          return (int) (date.getTime() / DateTimeUtils.MILLIS_PER_DAY);
//        } catch (ParseException e) {
//          return null;
//        }
      } else {
        throw new AssertionError("unrecognized type, value: "+string+", fieldtype: "+fieldType.getName());
      }
    }
  }

  /** Array row converter. */
  public static class ArrayRowConverter extends RowConverter<Object[]> {
    private final RelDataTypeField[] fieldTypes;
    private final int[] fields;
    // whether the row to convert is from a stream
    private final boolean stream;

    public ArrayRowConverter(List<RelDataTypeField> fieldTypes, int[] fields, boolean mock) {
      super(mock);
      this.fieldTypes = fieldTypes.toArray(new RelDataTypeField[fieldTypes.size()]);
      this.fields = fields;
      this.stream = false;
    }

//    ArrayRowConverter(List<RelDataTypeField> fieldTypes, int[] fields, boolean stream) {
//      this.fieldTypes = fieldTypes.toArray(new RelDataTypeField[fieldTypes.size()]);
//      this.fields = fields;
//      this.stream = stream;
//    }

    private static Class convertSqlTypeToJavaType(SqlTypeName sqlType) {
      if (sqlType == SqlTypeName.BOOLEAN) {
        return Boolean.class;
      } else if (sqlType == SqlTypeName.INTEGER) {
        return Integer.class;
      } else if (sqlType == SqlTypeName.BIGINT) {
        return Long.class;
      } else if (sqlType == SqlTypeName.FLOAT) {
        return Double.class;
      } else if (sqlType == SqlTypeName.DOUBLE) {
        return Double.class;
      } else if (sqlType == SqlTypeName.DECIMAL) {
        return BigDecimal.class;
      } else if (sqlType == SqlTypeName.VARCHAR) {
        return String.class;
      } else if (sqlType == SqlTypeName.DATE) {
        return Integer.class;
      } else {
        throw new AssertionError("unrecognized fieldtype: " + sqlType);
      }
    }

//    public static ArrayRowConverter<?> of(List<RelDataTypeField> fieldTypes){
//      ImmutableList.Builder builder = ImmutableList.builder();
//      for (RelDataTypeField f: fieldTypes) {
//        builder.add(convertSqlTypeToJavaType(f.getType().getSqlTypeName()));
//      }
//      return new ArrayRowConverter<builder.build()>()
//    }

    public Object[] convertRow(String[] strings) {
      if (mock) strings = new String[fields.length];
      if (stream) {
        return convertStreamRow(strings);
      } else {
        return convertNormalRow(strings);
      }
    }

    protected Object[] convertNormalRow(String[] strings) {
      final Object[] objects = new Object[fields.length];
      for (int i = 0; i < fields.length; i++) {
        int field = fields[i];
        objects[i] = convert(fieldTypes[field], strings[field]);
      }
      return objects;
    }

    protected Object[] convertStreamRow(String[] strings) {
      final Object[] objects = new Object[fields.length + 1];
      objects[0] = System.currentTimeMillis();
      for (int i = 0; i < fields.length; i++) {
        int field = fields[i];
        objects[i + 1] = convert(fieldTypes[field], strings[field]);
      }
      return objects;
    }
  }

  /** Single column row converter. */
  private static class SingleColumnRowConverter extends RowConverter {
    private final RelDataTypeField fieldType;
    private final int fieldIndex;

    private SingleColumnRowConverter(RelDataTypeField fieldType, int fieldIndex, boolean mock) {
      super(mock);
      this.fieldType = fieldType;
      this.fieldIndex = fieldIndex;
    }

    public Object convertRow(String[] strings) {
      return convert(fieldType, strings[fieldIndex]);
    }
  }
}

// End CsvEnumerator.java
