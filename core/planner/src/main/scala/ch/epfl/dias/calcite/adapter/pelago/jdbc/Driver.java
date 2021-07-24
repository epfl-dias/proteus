package ch.epfl.dias.calcite.adapter.pelago.jdbc;

import ch.epfl.dias.calcite.adapter.pelago.ddl.PelagoDdlExecutor;
import org.apache.calcite.avatica.*;
import org.apache.calcite.config.CalciteConnectionProperty;
import org.apache.calcite.config.Lex;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.PelagoMetaImpl;
import org.apache.calcite.linq4j.function.Function0;
import org.apache.calcite.prepare.PelagoPrepareImpl;

import java.sql.Connection;
import java.sql.SQLException;

public class Driver extends org.apache.calcite.jdbc.Driver {
    public static final String CONNECT_STRING_PREFIX = "jdbc:proteus:";

    static {
        new Driver().register();
    }

    @Override protected String getConnectStringPrefix() {
        return CONNECT_STRING_PREFIX;
    }

    protected DriverVersion createDriverVersion() {
        return DriverVersion.load(
                Driver.class,
                "ch-epfl-dias-pelago-jdbc.properties",
                "Proteus JDBC Driver",
                "unknown version",
                "Proteus",
                "unknown version");
    }

    protected Function0<CalcitePrepare> createPrepareFactory() {
        return PelagoPrepareImpl::new;
    }

    @Override public Meta createMeta(AvaticaConnection connection) {
        return new PelagoMetaImpl(connection);
    }

    public Connection connect(String url, java.util.Properties info) throws SQLException{
      info.put(CalciteConnectionProperty.LEX.name(), "Java");
      info.put(CalciteConnectionProperty.PARSER_FACTORY.name(), PelagoDdlExecutor.class.getName() + "#PARSER_FACTORY");
      return super.connect(url, info);
    }
}
