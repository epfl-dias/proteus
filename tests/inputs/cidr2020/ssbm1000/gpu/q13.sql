{
  "operator" : "print",
  "gpu" : false,
  "e" : [ {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "revenue",
        "relName" : "subsetPelagoDeviceCross#2143"
      } ],
      "type" : {
        "relName" : "subsetPelagoDeviceCross#2143",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "revenue",
      "relName" : "subsetPelagoDeviceCross#2143"
    },
    "register_as" : {
      "attrName" : "revenue",
      "relName" : "print2144"
    }
  } ],
  "input" : {
    "operator" : "gpu-to-cpu",
    "projections" : [ {
      "relName" : "subsetPelagoDeviceCross#2143",
      "attrName" : "revenue"
    } ],
    "queueSize" : 262144,
    "granularity" : "thread",
    "input" : {
      "operator" : "reduce",
      "gpu" : true,
      "e" : [ {
        "expression" : "recordProjection",
        "e" : {
          "expression" : "argument",
          "attributes" : [ {
            "attrName" : "revenue",
            "relName" : "subsetPelagoUnpack#2141"
          } ],
          "type" : {
            "relName" : "subsetPelagoUnpack#2141",
            "type" : "record"
          },
          "argNo" : -1
        },
        "attribute" : {
          "attrName" : "revenue",
          "relName" : "subsetPelagoUnpack#2141"
        },
        "register_as" : {
          "attrName" : "revenue",
          "relName" : "subsetPelagoDeviceCross#2143"
        }
      } ],
      "accumulator" : [ "sum" ],
      "p" : {
        "expression" : "bool",
        "v" : true
      },
      "input" : {
        "operator" : "unpack",
        "gpu" : true,
        "projections" : [ {
          "expression" : "recordProjection",
          "e" : {
            "expression" : "argument",
            "attributes" : [ {
              "attrName" : "revenue",
              "relName" : "subsetPelagoUnpack#2141"
            } ],
            "type" : {
              "relName" : "subsetPelagoUnpack#2141",
              "type" : "record"
            },
            "argNo" : -1
          },
          "attribute" : {
            "attrName" : "revenue",
            "relName" : "subsetPelagoUnpack#2141"
          }
        } ],
        "input" : {
          "operator" : "cpu-to-gpu",
          "projections" : [ {
            "relName" : "subsetPelagoUnpack#2141",
            "attrName" : "revenue",
            "isBlock" : true
          } ],
          "queueSize" : 262144,
          "granularity" : "thread",
          "input" : {
            "operator" : "mem-move-device",
            "projections" : [ {
              "relName" : "subsetPelagoUnpack#2141",
              "attrName" : "revenue",
              "isBlock" : true
            } ],
            "input" : {
              "operator" : "router",
              "gpu" : false,
              "projections" : [ {
                "relName" : "subsetPelagoUnpack#2141",
                "attrName" : "revenue",
                "isBlock" : true
              } ],
              "numOfParents" : 1,
              "producers" : 2,
              "slack" : 8,
              "cpu_targets" : false,
              "input" : {
                "operator" : "mem-move-device",
                "projections" : [ {
                  "relName" : "subsetPelagoUnpack#2141",
                  "attrName" : "revenue",
                  "isBlock" : true
                } ],
                "input" : {
                  "operator" : "gpu-to-cpu",
                  "projections" : [ {
                    "relName" : "subsetPelagoUnpack#2141",
                    "attrName" : "revenue",
                    "isBlock" : true
                  } ],
                  "queueSize" : 262144,
                  "granularity" : "thread",
                  "input" : {
                    "operator" : "pack",
                    "gpu" : true,
                    "projections" : [ {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "revenue",
                          "relName" : "subsetPelagoUnpack#2141"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#2141",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "revenue",
                        "relName" : "subsetPelagoUnpack#2141"
                      }
                    } ],
                    "input" : {
                      "operator" : "reduce",
                      "gpu" : true,
                      "e" : [ {
                        "expression" : "recordProjection",
                        "e" : {
                          "expression" : "argument",
                          "attributes" : [ {
                            "attrName" : "$f0",
                            "relName" : "subsetPelagoProject#2135"
                          } ],
                          "type" : {
                            "relName" : "subsetPelagoProject#2135",
                            "type" : "record"
                          },
                          "argNo" : -1
                        },
                        "attribute" : {
                          "attrName" : "$f0",
                          "relName" : "subsetPelagoProject#2135"
                        },
                        "register_as" : {
                          "attrName" : "revenue",
                          "relName" : "subsetPelagoUnpack#2141"
                        }
                      } ],
                      "accumulator" : [ "sum" ],
                      "p" : {
                        "expression" : "bool",
                        "v" : true
                      },
                      "input" : {
                        "operator" : "project",
                        "gpu" : true,
                        "relName" : "subsetPelagoProject#2135",
                        "e" : [ {
                          "expression" : "recordProjection",
                          "e" : {
                            "expression" : "argument",
                            "attributes" : [ {
                              "attrName" : "*",
                              "relName" : "subsetPelagoProject#2135"
                            } ],
                            "type" : {
                              "relName" : "subsetPelagoProject#2135",
                              "type" : "record"
                            },
                            "argNo" : -1
                          },
                          "attribute" : {
                            "attrName" : "*",
                            "relName" : "subsetPelagoProject#2135"
                          },
                          "register_as" : {
                            "attrName" : "$f0",
                            "relName" : "subsetPelagoProject#2135"
                          }
                        } ],
                        "input" : {
                          "operator" : "hashjoin-chained",
                          "gpu" : true,
                          "build_k" : {
                            "expression" : "recordProjection",
                            "e" : {
                              "expression" : "argument",
                              "attributes" : [ {
                                "attrName" : "d_datekey",
                                "relName" : "subsetPelagoProject#2128"
                              } ],
                              "type" : {
                                "relName" : "subsetPelagoProject#2128",
                                "type" : "record"
                              },
                              "argNo" : -1
                            },
                            "attribute" : {
                              "attrName" : "d_datekey",
                              "relName" : "subsetPelagoProject#2128"
                            },
                            "register_as" : {
                              "attrName" : "$1",
                              "relName" : "subsetPelagoProject#2135"
                            }
                          },
                          "build_e" : [ {
                            "e" : {
                              "expression" : "recordProjection",
                              "e" : {
                                "expression" : "argument",
                                "attributes" : [ {
                                  "attrName" : "d_datekey",
                                  "relName" : "subsetPelagoProject#2128"
                                } ],
                                "type" : {
                                  "relName" : "subsetPelagoProject#2128",
                                  "type" : "record"
                                },
                                "argNo" : -1
                              },
                              "attribute" : {
                                "attrName" : "d_datekey",
                                "relName" : "subsetPelagoProject#2128"
                              },
                              "register_as" : {
                                "attrName" : "d_datekey",
                                "relName" : "subsetPelagoProject#2135"
                              }
                            },
                            "packet" : 1,
                            "offset" : 0
                          } ],
                          "build_w" : [ 64, 32 ],
                          "build_input" : {
                            "operator" : "project",
                            "gpu" : true,
                            "relName" : "subsetPelagoProject#2128",
                            "e" : [ {
                              "expression" : "recordProjection",
                              "e" : {
                                "expression" : "argument",
                                "attributes" : [ {
                                  "attrName" : "d_datekey",
                                  "relName" : "inputs/ssbm1000/date.csv"
                                } ],
                                "type" : {
                                  "relName" : "inputs/ssbm1000/date.csv",
                                  "type" : "record"
                                },
                                "argNo" : -1
                              },
                              "attribute" : {
                                "attrName" : "d_datekey",
                                "relName" : "inputs/ssbm1000/date.csv"
                              },
                              "register_as" : {
                                "attrName" : "d_datekey",
                                "relName" : "subsetPelagoProject#2128"
                              }
                            } ],
                            "input" : {
                              "operator" : "select",
                              "gpu" : true,
                              "p" : {
                                "expression" : "and",
                                "left" : {
                                  "expression" : "eq",
                                  "left" : {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "d_weeknuminyear",
                                        "relName" : "inputs/ssbm1000/date.csv"
                                      } ],
                                      "type" : {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "d_weeknuminyear",
                                      "relName" : "inputs/ssbm1000/date.csv"
                                    }
                                  },
                                  "right" : {
                                    "expression" : "int",
                                    "v" : 6
                                  }
                                },
                                "right" : {
                                  "expression" : "eq",
                                  "left" : {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "d_year",
                                        "relName" : "inputs/ssbm1000/date.csv"
                                      } ],
                                      "type" : {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "d_year",
                                      "relName" : "inputs/ssbm1000/date.csv"
                                    }
                                  },
                                  "right" : {
                                    "expression" : "int",
                                    "v" : 1994
                                  }
                                }
                              },
                              "input" : {
                                "operator" : "unpack",
                                "gpu" : true,
                                "projections" : [ {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "d_datekey",
                                      "relName" : "inputs/ssbm1000/date.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "d_datekey",
                                    "relName" : "inputs/ssbm1000/date.csv"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "d_year",
                                      "relName" : "inputs/ssbm1000/date.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "d_year",
                                    "relName" : "inputs/ssbm1000/date.csv"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "d_weeknuminyear",
                                      "relName" : "inputs/ssbm1000/date.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "d_weeknuminyear",
                                    "relName" : "inputs/ssbm1000/date.csv"
                                  }
                                } ],
                                "input" : {
                                  "operator" : "cpu-to-gpu",
                                  "projections" : [ {
                                    "relName" : "inputs/ssbm1000/date.csv",
                                    "attrName" : "d_datekey",
                                    "isBlock" : true
                                  }, {
                                    "relName" : "inputs/ssbm1000/date.csv",
                                    "attrName" : "d_year",
                                    "isBlock" : true
                                  }, {
                                    "relName" : "inputs/ssbm1000/date.csv",
                                    "attrName" : "d_weeknuminyear",
                                    "isBlock" : true
                                  } ],
                                  "queueSize" : 262144,
                                  "granularity" : "thread",
                                  "input" : {
                                    "operator" : "router",
                                    "gpu" : false,
                                    "projections" : [ {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "attrName" : "d_datekey",
                                      "isBlock" : true
                                    }, {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "attrName" : "d_year",
                                      "isBlock" : true
                                    }, {
                                      "relName" : "inputs/ssbm1000/date.csv",
                                      "attrName" : "d_weeknuminyear",
                                      "isBlock" : true
                                    } ],
                                    "numOfParents" : 2,
                                    "producers" : 1,
                                    "slack" : 8,
                                    "cpu_targets" : false,
                                    "target" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "argNo" : -1,
                                        "type" : {
                                          "type" : "record",
                                          "relName" : "inputs/ssbm1000/date.csv"
                                        },
                                        "attributes" : [ {
                                          "relName" : "inputs/ssbm1000/date.csv",
                                          "attrName" : "__broadcastTarget"
                                        } ]
                                      },
                                      "attribute" : {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "attrName" : "__broadcastTarget"
                                      }
                                    },
                                    "input" : {
                                      "operator" : "mem-broadcast-device",
                                      "num_of_targets" : 2,
                                      "projections" : [ {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "attrName" : "d_datekey",
                                        "isBlock" : true
                                      }, {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "attrName" : "d_year",
                                        "isBlock" : true
                                      }, {
                                        "relName" : "inputs/ssbm1000/date.csv",
                                        "attrName" : "d_weeknuminyear",
                                        "isBlock" : true
                                      } ],
                                      "input" : {
                                        "operator" : "scan",
                                        "gpu" : false,
                                        "plugin" : {
                                          "type" : "block",
                                          "linehint" : 2556,
                                          "name" : "inputs/ssbm1000/date.csv",
                                          "projections" : [ {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_datekey"
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_year"
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_weeknuminyear"
                                          } ],
                                          "schema" : [ {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_datekey",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 1
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_date",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 2
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_dayofweek",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 3
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_month",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 4
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_year",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 5
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_yearmonthnum",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 6
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_yearmonth",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 7
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_daynuminweek",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 8
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_daynuminmonth",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 9
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_daynuminyear",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 10
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_monthnuminyear",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 11
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_weeknuminyear",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 12
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_sellingseason",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 13
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_lastdayinweekfl",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 14
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_lastdayinmonthfl",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 15
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_holidayfl",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 16
                                          }, {
                                            "relName" : "inputs/ssbm1000/date.csv",
                                            "attrName" : "d_weekdayfl",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 17
                                          } ]
                                        }
                                      },
                                      "to_cpu" : false
                                    }
                                  }
                                }
                              }
                            }
                          },
                          "probe_k" : {
                            "expression" : "recordProjection",
                            "e" : {
                              "expression" : "argument",
                              "attributes" : [ {
                                "attrName" : "lo_orderdate",
                                "relName" : "subsetPelagoProject#2133"
                              } ],
                              "type" : {
                                "relName" : "subsetPelagoProject#2133",
                                "type" : "record"
                              },
                              "argNo" : -1
                            },
                            "attribute" : {
                              "attrName" : "lo_orderdate",
                              "relName" : "subsetPelagoProject#2133"
                            },
                            "register_as" : {
                              "attrName" : "$0",
                              "relName" : "subsetPelagoProject#2135"
                            }
                          },
                          "probe_e" : [ {
                            "e" : {
                              "expression" : "recordProjection",
                              "e" : {
                                "expression" : "argument",
                                "attributes" : [ {
                                  "attrName" : "lo_orderdate",
                                  "relName" : "subsetPelagoProject#2133"
                                } ],
                                "type" : {
                                  "relName" : "subsetPelagoProject#2133",
                                  "type" : "record"
                                },
                                "argNo" : -1
                              },
                              "attribute" : {
                                "attrName" : "lo_orderdate",
                                "relName" : "subsetPelagoProject#2133"
                              },
                              "register_as" : {
                                "attrName" : "lo_orderdate",
                                "relName" : "subsetPelagoProject#2135"
                              }
                            },
                            "packet" : 1,
                            "offset" : 0
                          }, {
                            "e" : {
                              "expression" : "recordProjection",
                              "e" : {
                                "expression" : "argument",
                                "attributes" : [ {
                                  "attrName" : "*",
                                  "relName" : "subsetPelagoProject#2133"
                                } ],
                                "type" : {
                                  "relName" : "subsetPelagoProject#2133",
                                  "type" : "record"
                                },
                                "argNo" : -1
                              },
                              "attribute" : {
                                "attrName" : "*",
                                "relName" : "subsetPelagoProject#2133"
                              },
                              "register_as" : {
                                "attrName" : "*",
                                "relName" : "subsetPelagoProject#2135"
                              }
                            },
                            "packet" : 2,
                            "offset" : 0
                          } ],
                          "probe_w" : [ 64, 32, 32 ],
                          "hash_bits" : 10,
                          "maxBuildInputSize" : 2556,
                          "probe_input" : {
                            "operator" : "project",
                            "gpu" : true,
                            "relName" : "subsetPelagoProject#2133",
                            "e" : [ {
                              "expression" : "recordProjection",
                              "e" : {
                                "expression" : "argument",
                                "attributes" : [ {
                                  "attrName" : "lo_orderdate",
                                  "relName" : "inputs/ssbm1000/lineorder.csv"
                                } ],
                                "type" : {
                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                  "type" : "record"
                                },
                                "argNo" : -1
                              },
                              "attribute" : {
                                "attrName" : "lo_orderdate",
                                "relName" : "inputs/ssbm1000/lineorder.csv"
                              },
                              "register_as" : {
                                "attrName" : "lo_orderdate",
                                "relName" : "subsetPelagoProject#2133"
                              }
                            }, {
                              "expression" : "multiply",
                              "left" : {
                                "expression" : "recordProjection",
                                "e" : {
                                  "expression" : "argument",
                                  "attributes" : [ {
                                    "attrName" : "lo_extendedprice",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  } ],
                                  "type" : {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "type" : "record"
                                  },
                                  "argNo" : -1
                                },
                                "attribute" : {
                                  "attrName" : "lo_extendedprice",
                                  "relName" : "inputs/ssbm1000/lineorder.csv"
                                }
                              },
                              "right" : {
                                "expression" : "recordProjection",
                                "e" : {
                                  "expression" : "argument",
                                  "attributes" : [ {
                                    "attrName" : "lo_discount",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  } ],
                                  "type" : {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "type" : "record"
                                  },
                                  "argNo" : -1
                                },
                                "attribute" : {
                                  "attrName" : "lo_discount",
                                  "relName" : "inputs/ssbm1000/lineorder.csv"
                                }
                              },
                              "register_as" : {
                                "attrName" : "*",
                                "relName" : "subsetPelagoProject#2133"
                              }
                            } ],
                            "input" : {
                              "operator" : "select",
                              "gpu" : true,
                              "p" : {
                                "expression" : "and",
                                "left" : {
                                  "expression" : "and",
                                  "left" : {
                                    "expression" : "ge",
                                    "left" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "lo_discount",
                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                        } ],
                                        "type" : {
                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "lo_discount",
                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                      }
                                    },
                                    "right" : {
                                      "expression" : "int",
                                      "v" : 5
                                    }
                                  },
                                  "right" : {
                                    "expression" : "le",
                                    "left" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "lo_discount",
                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                        } ],
                                        "type" : {
                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "lo_discount",
                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                      }
                                    },
                                    "right" : {
                                      "expression" : "int",
                                      "v" : 7
                                    }
                                  }
                                },
                                "right" : {
                                  "expression" : "and",
                                  "left" : {
                                    "expression" : "ge",
                                    "left" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "lo_quantity",
                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                        } ],
                                        "type" : {
                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "lo_quantity",
                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                      }
                                    },
                                    "right" : {
                                      "expression" : "int",
                                      "v" : 26
                                    }
                                  },
                                  "right" : {
                                    "expression" : "le",
                                    "left" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "lo_quantity",
                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                        } ],
                                        "type" : {
                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "lo_quantity",
                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                      }
                                    },
                                    "right" : {
                                      "expression" : "int",
                                      "v" : 35
                                    }
                                  }
                                }
                              },
                              "input" : {
                                "operator" : "unpack",
                                "gpu" : true,
                                "projections" : [ {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "lo_orderdate",
                                      "relName" : "inputs/ssbm1000/lineorder.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "lo_orderdate",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "lo_quantity",
                                      "relName" : "inputs/ssbm1000/lineorder.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "lo_quantity",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "lo_extendedprice",
                                      "relName" : "inputs/ssbm1000/lineorder.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "lo_extendedprice",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "lo_discount",
                                      "relName" : "inputs/ssbm1000/lineorder.csv"
                                    } ],
                                    "type" : {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "lo_discount",
                                    "relName" : "inputs/ssbm1000/lineorder.csv"
                                  }
                                } ],
                                "input" : {
                                  "operator" : "cpu-to-gpu",
                                  "projections" : [ {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "attrName" : "lo_orderdate",
                                    "isBlock" : true
                                  }, {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "attrName" : "lo_quantity",
                                    "isBlock" : true
                                  }, {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "attrName" : "lo_extendedprice",
                                    "isBlock" : true
                                  }, {
                                    "relName" : "inputs/ssbm1000/lineorder.csv",
                                    "attrName" : "lo_discount",
                                    "isBlock" : true
                                  } ],
                                  "queueSize" : 262144,
                                  "granularity" : "thread",
                                  "input" : {
                                    "operator" : "mem-move-device",
                                    "projections" : [ {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "attrName" : "lo_orderdate",
                                      "isBlock" : true
                                    }, {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "attrName" : "lo_quantity",
                                      "isBlock" : true
                                    }, {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "attrName" : "lo_extendedprice",
                                      "isBlock" : true
                                    }, {
                                      "relName" : "inputs/ssbm1000/lineorder.csv",
                                      "attrName" : "lo_discount",
                                      "isBlock" : true
                                    } ],
                                    "input" : {
                                      "operator" : "router",
                                      "gpu" : false,
                                      "projections" : [ {
                                        "relName" : "inputs/ssbm1000/lineorder.csv",
                                        "attrName" : "lo_orderdate",
                                        "isBlock" : true
                                      }, {
                                        "relName" : "inputs/ssbm1000/lineorder.csv",
                                        "attrName" : "lo_quantity",
                                        "isBlock" : true
                                      }, {
                                        "relName" : "inputs/ssbm1000/lineorder.csv",
                                        "attrName" : "lo_extendedprice",
                                        "isBlock" : true
                                      }, {
                                        "relName" : "inputs/ssbm1000/lineorder.csv",
                                        "attrName" : "lo_discount",
                                        "isBlock" : true
                                      } ],
                                      "numOfParents" : 2,
                                      "producers" : 1,
                                      "slack" : 8,
                                      "cpu_targets" : false,
                                      "numa_local" : true,
                                      "input" : {
                                        "operator" : "scan",
                                        "gpu" : false,
                                        "plugin" : {
                                          "type" : "block",
                                          "linehint" : 5999989813,
                                          "name" : "inputs/ssbm1000/lineorder.csv",
                                          "projections" : [ {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_orderdate"
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_quantity"
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_extendedprice"
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_discount"
                                          } ],
                                          "schema" : [ {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_orderkey",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 1
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_linenumber",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 2
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_custkey",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 3
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_partkey",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 4
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_suppkey",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 5
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_orderdate",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 6
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_orderpriority",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 7
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_shippriority",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 8
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_quantity",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 9
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_extendedprice",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 10
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_ordtotalprice",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 11
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_discount",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 12
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_revenue",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 13
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_supplycost",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 14
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_tax",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 15
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_commitdate",
                                            "type" : {
                                              "type" : "int"
                                            },
                                            "attrNo" : 16
                                          }, {
                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                            "attrName" : "lo_shipmode",
                                            "type" : {
                                              "type" : "dstring"
                                            },
                                            "attrNo" : 17
                                          } ]
                                        }
                                      }
                                    },
                                    "to_cpu" : false
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    },
                    "trait" : "Pelago.[].packed.NVPTX.homRandom.hetSingle"
                  }
                },
                "to_cpu" : false
              }
            },
            "to_cpu" : false
          }
        }
      }
    }
  }
}