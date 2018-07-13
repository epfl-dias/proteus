// Updated license information

/**
 * Parse tree for SQL DDL statements.
 *
 * <p>These are available in the extended SQL parser that is part of Calcite's
 * "server" module; the core parser in the "core" module only supports SELECT
 * and DML.
 *
 * <p>If you are writing a project that requires DDL it is likely that your
 * DDL syntax is different than ours. We recommend that you copy-paste this
 * the parser and its supporting classes into your own module, rather than try
 * to extend this one.
 */
@PackageMarker
package org.apache.calcite.sql.ddl;

import org.apache.calcite.avatica.util.PackageMarker;

// End package-info.java
