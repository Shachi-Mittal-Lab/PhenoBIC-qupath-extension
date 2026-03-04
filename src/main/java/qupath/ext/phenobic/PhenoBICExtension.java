package qupath.ext.phenobic;

import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.scripting.ScriptEditor;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * QuPath extension for PhenoBIC phenotype inference.
 * Adds a "Run PhenoBIC" submenu with: Cell phenotyping with PhenoBIC, Print Channel Names.
 * Each option opens the corresponding Groovy script in the script editor.
 */
public class PhenoBICExtension implements QuPathExtension {

    private static final Logger logger = LoggerFactory.getLogger(PhenoBICExtension.class);
    private static final String EXTENSION_NAME = "PhenoBIC";
    private static final String SCRIPT_PHENOBIC = "/scripts/PhenoBIC_Run.groovy";
    private static final String SCRIPT_PRINT_CHANNELS = "/scripts/print_channels.groovy";
    private static final Version QUPATH_VERSION = Version.parse("v0.6.0");

    private boolean installed = false;

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (installed) {
            logger.debug("{} already installed", getName());
            return;
        }
        installed = true;
        var extMenu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);

        Menu runPhenoBICMenu = new Menu("Run PhenoBIC");

        MenuItem cellPhenotyping = new MenuItem("Cell phenotyping with PhenoBIC");
        cellPhenotyping.setOnAction(e -> openScript(qupath, SCRIPT_PHENOBIC, "PhenoBIC_Run", ".groovy"));

        MenuItem printChannels = new MenuItem("Print Channel Names");
        printChannels.setOnAction(e -> openScript(qupath, SCRIPT_PRINT_CHANNELS, "print_channels", ".groovy"));

        runPhenoBICMenu.getItems().addAll(cellPhenotyping, printChannels);
        extMenu.getItems().add(runPhenoBICMenu);
    }

    private static void openScript(QuPathGUI qupath, String resourcePath, String tempPrefix, String tempSuffix) {
        try (var in = PhenoBICExtension.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                logger.error("Script not found: {}", resourcePath);
                return;
            }
            String script = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))
                    .lines()
                    .collect(Collectors.joining("\n"));
            File tempFile = File.createTempFile(tempPrefix, tempSuffix);
            tempFile.deleteOnExit();
            try (PrintWriter w = new PrintWriter(tempFile, StandardCharsets.UTF_8)) {
                w.print(script);
            }
            ScriptEditor editor = qupath.getScriptEditor();
            if (editor != null) {
                editor.showScript(tempFile);
                editor.showEditor();
            }
        } catch (Exception ex) {
            logger.error("Failed to open script: {}", resourcePath, ex);
        }
    }

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return "Run PhenoBIC: cell phenotyping with PhenoBIC or print channel names. Scripts open in the script editor; edit CONFIG then Run or Run for project.";
    }

    @Override
    public Version getQuPathVersion() {
        return QUPATH_VERSION;
    }
}
