context("test-mock")

test_that("Initial test works", {
  lf <- lazy_frame(x = 1, src = simulate_vidar())

  expect_match(lf %>% head() %>% sql_render(simulate_vidar()),
               sql("SELECT [*] FROM \"df\""))
})
